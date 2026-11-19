"""HDO screen: effect of deuterated water on the deep Venus column (surface to 1.05 bar),
monochromatic line-by-line, 900-9600 cm-1. HITRAN2020 CO2, H2O, SO2 lines; CO2-CO2 CIA
from the given file (negatives zeroed, overlapping blocks interpolated in T as one series),
no near-IR window CIA. Deuterated isotopologues (4, 5, 6) scaled
by f relative to terrestrial, D2O (7) by f**2.
Usage: python3 -I hdo_screen.py <hitran_dir> <parsum.dat> <cia file>
"""
import sys
import numpy as np

hdir, parsum_f, cia_f = sys.argv[1:4]
C2, KB, AMU, G, M_AIR = 1.4387769, 1.380649e-23, 1.66053907e-27, 8.87, 43.4
NU0, NU1, DNU = 900.0, 9600.0, 0.05
nu = np.arange(NU0, NU1 + DNU / 2, DNU)

# venus_atm_obs_so2.raw levels below 1.05 bar (Pa, K); H2O mass ratio per level
p_lev = np.array([1.05e5, 1.46e5, 2.04e5, 2.84e5, 3.97e5, 5.54e5, 7.73e5, 1.08e6, 1.51e6,
                  2.07e6, 2.76e6, 3.52e6, 4.35e6, 5.19e6, 6.01e6, 6.77e6, 7.44e6, 8.01e6,
                  8.45e6, 8.77e6, 8.98e6, 9.11e6, 9.19e6])
t_lev = np.array([341, 362, 380, 400, 424, 451, 481, 512, 543, 575, 603, 628, 649, 668, 684,
                  697, 708, 716, 723, 727, 729, 730, 731.0])
x_co2 = 0.977 * M_AIR / 44.01
x_so2 = 1.5e-4 * M_AIR / 64.07
x_h2o = 1.2e-5 * M_AIR / 18.02
edges = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 22]   # merge profile levels into 10 layers
layers = []
for a, b in zip(edges[:-1], edges[1:]):
    pt, pb = p_lev[a], p_lev[b]
    p = np.sqrt(pt * pb)
    t = np.interp(np.log(p), np.log(p_lev), t_lev)
    layers.append(dict(p=p, t=t, ncol=(pb - pt) / (M_AIR * AMU * G) * 1e-4,
                       dz=KB * t / (M_AIR * AMU * G) * np.log(pb / pt) * 100.0))
T_SURF = t_lev[-1]

hdr = open(parsum_f).readline().split()
cols = ["Temp(K)", "H2O_1", "H2O_2", "H2O_3", "H2O_4", "H2O_5", "H2O_6", "CO2_7", "SO2_42"]
idx = [hdr.index(c) for c in cols]
rows = []
with open(parsum_f) as f:
    next(f)
    for l in f:
        w = l.split()
        if float(w[0]) > 1000:
            break
        rows.append([float(w[j]) for j in idx])
ps = np.array(rows)
def qratio(col, t):
    j = cols.index(col)
    return np.interp(296.0, ps[:, 0], ps[:, j]) / np.interp(t, ps[:, 0], ps[:, j])

def read_lines(path, offset, smin, lo, hi):
    out = []
    with open(path) as f:
        for l in f:
            o = offset
            v = float(l[3 + o:15 + o])
            if v < lo or v > hi:
                continue
            s = float(l[15 + o:25 + o])
            if s < smin:
                continue
            iso = int(l[2]) if offset == 0 else int(l[3:6])
            if offset == 0:
                ga, gs, e, n = float(l[35:40]), float(l[40:45]), float(l[45:55]), float(l[55:59])
            else:
                ga, gs, e, n = float(l[38:44]), float(l[44:49]), float(l[49:59]), float(l[59:65])
            out.append((v, s, ga, gs, e, n, iso))
    return np.array(out)

def p_h_chi(d, t):
    tc = min(800.0, max(190.0, t))
    b1 = 0.0888 - 0.16 * np.exp(-0.0041 * tc)
    b2 = 0.0526 * np.exp(-0.00152 * tc)
    b3 = 0.0232
    d = np.abs(d)
    chi = np.ones_like(d)
    m = (d >= 3) & (d < 30); chi[m] = np.exp(-b1 * (d[m] - 3))
    m = (d >= 30) & (d < 120); chi[m] = np.exp(-b1 * 27 - b2 * (d[m] - 30))
    m = d >= 120; chi[m] = np.exp(-b1 * 27 - b2 * 90 - b3 * (d[m] - 120))
    return chi

def k_lines(lines, t, p_atm, qcols, gidx, gscale, cut, chi):
    v, s, e, n, iso = lines[:, 0], lines[:, 1], lines[:, 4], lines[:, 5], lines[:, 6].astype(int)
    q = np.array([qratio(qcols.get(i, qcols["default"]), t) for i in iso])
    st = (s * q * np.exp(-C2 * e * (1 / t - 1 / 296.0))
          * (1 - np.exp(-C2 * v / t)) / (1 - np.exp(-C2 * v / 296.0)))
    gam = lines[:, gidx] * gscale * p_atm * (296.0 / t) ** n
    k = np.zeros_like(nu)
    nwin = int(round(cut / DNU))
    for vi, si, gi in zip(v, st, gam):
        c = int(round((vi - NU0) / DNU))
        a, b = max(c - nwin, 0), min(c + nwin + 1, nu.size)
        if a >= b:
            continue
        d = nu[a:b] - vi
        prof = gi / np.pi / (d * d + gi * gi)
        if chi:
            prof = prof * p_h_chi(d, t)
        k[a:b] += si * prof
    return k

def read_cia(path):
    """Blocks as (temperature, values on the nu grid, NaN outside the block). Negative values
    are set to zero, as SOCRATES's access_cia_int does."""
    blocks, lines = [], open(path).readlines()
    i = 0
    while i < len(lines):
        h = lines[i].split(); npt = int(h[3])
        d = np.array([[float(x) for x in lines[i + 1 + j].split()[:2]] for j in range(npt)])
        y = np.interp(nu, d[:, 0], np.maximum(d[:, 1], 0.0), left=np.nan, right=np.nan)
        blocks.append((float(h[4]), y)); i += npt + 1
    return np.array([b[0] for b in blocks]), np.array([b[1] for b in blocks])

def k_cia(blocks, t):
    """At each wavenumber, interpolate linearly in T between the blocks that cover it, holding
    the nearest tabulated temperature outside their range. Blocks from different sources that
    cover the same wavenumber form one temperature series; they are never summed."""
    temps, vals = blocks
    cover = ~np.isnan(vals)
    tt = temps[:, None] * np.ones_like(vals)
    below = np.where(cover & (tt <= t), tt, -np.inf)
    above = np.where(cover & (tt >= t), tt, np.inf)
    ib, ia = below.argmax(axis=0), above.argmin(axis=0)
    tb, ta = below.max(axis=0), above.min(axis=0)
    col = np.arange(nu.size)
    has_b, has_a = np.isfinite(tb), np.isfinite(ta)
    ib = np.where(has_b, ib, ia); ia = np.where(has_a, ia, ib)
    tb = np.where(has_b, tb, ta); ta = np.where(has_a, ta, tb)
    with np.errstate(invalid="ignore"):
        w = np.where(ta > tb, (t - tb) / np.where(ta > tb, ta - tb, 1.0), 0.0)
    k = (1 - w) * vals[ib, col] + w * vals[ia, col]
    return np.where(cover.any(axis=0), k, 0.0)

co2 = read_lines(f"{hdir}/HITRAN2020_CO2.par", 0, 1e-30, NU0 - 100, NU1 + 100)
so2 = read_lines(f"{hdir}/HITRAN2020_SO2.par_co2", 3, 1e-26, NU0 - 25, NU1 + 25)
h2o = read_lines(f"{hdir}/HITRAN2020_H2O.par", 0, 0.0, NU0 - 25, NU1 + 25)
deut = np.isin(h2o[:, 6], [4, 5, 6, 7])
print(f"lines: CO2 {len(co2)}, SO2 {len(so2)}, H2O main {np.sum(~deut)}, deuterated {np.sum(deut)}")
h2o_main, h2o_d1, h2o_d2 = h2o[~deut], h2o[np.isin(h2o[:, 6], [4, 5, 6])], h2o[h2o[:, 6] == 7]
cia = read_cia(cia_f)
qc_h2o = {1: "H2O_1", 2: "H2O_2", 3: "H2O_3", 4: "H2O_4", 5: "H2O_5", 6: "H2O_6", "default": "H2O_4"}
for L in layers:
    pa, t = L["p"] / 101325, L["t"]
    nco2 = x_co2 * L["p"] / (KB * t) * 1e-6
    L["bg"] = (k_lines(co2, t, pa, {"default": "CO2_7"}, 3, 1.0, 100.0, True) * L["ncol"] * x_co2
               + k_lines(so2, t, pa, {"default": "SO2_42"}, 2, 1.0, 25.0, False) * L["ncol"] * x_so2
               + k_cia(cia, t) * nco2 ** 2 * L["dz"])
    L["h2o"] = k_lines(h2o_main, t, pa, qc_h2o, 2, 1.5, 25.0, False) * L["ncol"] * x_h2o
    L["d1"] = k_lines(h2o_d1, t, pa, qc_h2o, 2, 1.5, 25.0, False) * L["ncol"] * x_h2o
    L["d2"] = (k_lines(h2o_d2, t, pa, qc_h2o, 2, 1.5, 25.0, False) * L["ncol"] * x_h2o
               if len(h2o_d2) else np.zeros_like(nu))
    print(f"layer p={L['p']/1e5:6.2f} bar T={t:5.0f} K done", flush=True)

def planck(t):
    return nu ** 3 / (np.exp(C2 * nu / t) - 1)

MU = 1 / 1.66
def fluxes(f):
    """Upward flux at the column top (thermal) and solar transmission to the surface."""
    taus = [(L["bg"] + L["h2o"] + f * L["d1"] + f * f * L["d2"]) / MU for L in layers]
    up = planck(T_SURF)                       # black surface
    for L, tau in zip(reversed(layers), reversed(taus)):
        tr = np.exp(-tau)
        up = up * tr + planck(L["t"]) * (1 - tr)
    sol = np.exp(-np.sum(taus, axis=0))
    return up, sol

cases = {"no HDO": 0.0, "x46": 46.0, "x127": 127.0}
res = {k: fluxes(f) for k, f in cases.items()}
sun = planck(5778.0)
windows = [(1000, 1300), (1300, 1600), (1600, 2200), (2200, 2500), (2500, 2900), (2900, 3500),
           (3500, 4150), (4150, 4550), (4550, 5500), (5500, 6000), (6000, 7500), (7500, 8000),
           (8000, 8700), (8700, 9300), (9300, 9600)]
print("\nRange (cm-1)   share of OLR(no HDO)  OLR change x46  x127 | share of surface solar  change x46  x127")
up0, so0 = res["no HDO"]
tot_up, tot_so = up0.sum(), np.sum(sun * so0)
for lo, hi in windows + [(900, 9600)]:
    m = (nu >= lo) & (nu < hi)
    u0, s0 = up0[m].sum(), np.sum(sun[m] * so0[m])
    du = [res[k][0][m].sum() / u0 - 1 for k in ("x46", "x127")]
    ds = [np.sum(sun[m] * res[k][1][m]) / s0 - 1 if s0 > 0 else 0 for k in ("x46", "x127")]
    print(f"{lo:5d}-{hi:5d}      {u0/tot_up:6.3f}           {du[0]:+7.2%} {du[1]:+7.2%} |"
          f"        {s0/tot_so:6.3f}         {ds[0]:+7.2%} {ds[1]:+7.2%}")
print("\n50 cm-1 bins where x127 changes OLR or surface sunlight by more than 2 % (and bin share > 0.5 %):")
for b in np.arange(NU0, NU1, 50):
    m = (nu >= b) & (nu < b + 50)
    u0, s0 = up0[m].sum(), np.sum(sun[m] * so0[m])
    du = res["x127"][0][m].sum() / u0 - 1
    ds = np.sum(sun[m] * res["x127"][1][m]) / s0 - 1 if s0 > 0 else 0
    if (abs(du) > 0.02 and u0 / tot_up > 0.005) or (abs(ds) > 0.02 and s0 / tot_so > 0.005):
        print(f"  {b:.0f}-{b+50:.0f} ({1e4/(b+25):.2f} um): OLR share {u0/tot_up:.3f}, change {du:+.1%};"
              f" solar share {s0/tot_so:.3f}, change {ds:+.1%}")
