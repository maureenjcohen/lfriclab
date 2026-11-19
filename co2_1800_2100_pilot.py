"""Mid-IR pilot: does filtered HITEMP 2024 CO2 change fluxes in 1800-2100 cm-1, where the recipe
uses HITRAN2020 because summed line strengths differ by only 2-7 %?
Monochromatic, 1800-2100 cm-1, column from ~100 Pa to the surface (venus_atm_obs_so2.raw),
no scattering, diffusivity 1.66, black surface. CO2 lines from 1700-2200 cm-1 (100 cm-1
cut-off), Perrin & Hartmann chi, as the recipe. CO2-CO2 CIA from the given file (negatives
zeroed; covering blocks interpolated in T as one series). HITRAN2020 H2O and SO2 lines.
Cases: CIA+H2O+SO2 (no CO2 lines), + HITRAN2020 CO2 (all lines), + filtered HITEMP CO2.

Line absorption is computed by depositing line strengths on the grid and convolving with the
chi-weighted Lorentz profile of each width class (classes 0.02 dex wide, so widths are within
about 2.5 % of the class value). Lines narrower than the grid step are widened to it with area
kept, so only differences between cases are meaningful in the upper atmosphere.
Usage: python3 -I co2_1800_2100_pilot.py <raw profile> <hitran dir> <hitemp co2 1700-2200 .par>
                  <parsum> <CO2-CO2 cia>
"""
import sys
import numpy as np
from scipy.signal import fftconvolve

raw, hdir, hitemp_f, parsum_f, cia_f = sys.argv[1:6]
C2, KB, AMU, G, M_AIR = 1.4387769, 1.380649e-23, 1.66053907e-27, 8.87, 43.4
C1 = 1.191042e-8                    # W m-2 sr-1 (cm-1)-4
CP = 1000.0                         # J kg-1 K-1, for heating rates only
LO, HI = 1800.0, 2100.0             # flux range
GLO, GHI, DNU = 1700.0, 2200.0, 0.005
CUT = 100.0
grid = np.arange(GLO, GHI + DNU / 2, DNU)
inr = (grid >= LO) & (grid < HI)
nu = grid[inr]

prof = []
for l in open(raw).readlines()[2:]:
    w = l.split()
    if len(w) >= 15 and float(w[0]) >= 100:
        prof.append([float(w[0]), float(w[1]), float(w[2]), float(w[3]), float(w[4])])
prof = np.array(prof)
p_lev, t_lev = prof[:, 0], prof[:, 1]
x_co2 = prof[:, 2] * M_AIR / 44.01
x_h2o = prof[:, 3] * M_AIR / 18.02
x_so2 = prof[:, 4] * M_AIR / 64.07
layers = []
for i in range(len(p_lev) - 1):
    pt, pb = p_lev[i], p_lev[i + 1]
    p = np.sqrt(pt * pb)
    w = lambda a: 0.5 * (a[i] + a[i + 1])
    t = w(t_lev)
    layers.append(dict(p=p, t=t, pt=pt, pb=pb, ncol=(pb - pt) / (M_AIR * AMU * G) * 1e-4,
                       dz=KB * t / (M_AIR * AMU * G) * np.log(pb / pt) * 100.0,
                       xco2=w(x_co2), xh2o=w(x_h2o), xso2=w(x_so2)))
T_SURF = t_lev[-1]

# Q(T) per isotopologue: HITRAN codes 1-9, 0 (CO2); 1-6 (H2O); 1 (SO2)
hdr = open(parsum_f).readline().split()
rows = []
with open(parsum_f) as f:
    next(f)
    for l in f:
        w = l.split()
        if float(w[0]) > 1000:
            break
        rows.append([float(x) for x in w])
tab = np.array(rows)
QCOL = {"CO2": ["CO2_7", "CO2_8", "CO2_9", "CO2_10", "CO2_11", "CO2_12", "CO2_13", "CO2_14",
                "CO2_121", "CO2_15"],
        "H2O": ["H2O_1", "H2O_2", "H2O_3", "H2O_4", "H2O_5", "H2O_6", "H2O_129"],
        "SO2": ["SO2_42", "SO2_43", "SO2_137", "SO2_138"]}
def qratio(gas, iso, t):
    """Q(296)/Q(t); iso is the 0-based position in QCOL[gas]."""
    out = np.full(iso.shape, np.nan)
    for k, c in enumerate(QCOL[gas]):
        if c not in hdr:
            continue
        j = hdr.index(c)
        out[iso == k] = np.interp(296.0, tab[:, 0], tab[:, j]) / np.interp(t, tab[:, 0], tab[:, j])
    return out

ISO = {"CO2": b"1234567890", "H2O": b"1234567", "SO2": b"1234"}
def read_lines(path, gas, offset, smin, lo, hi):
    """(v, S296, gamma_air, gamma_self, E'', n_air, iso index); offset 3 for .par_co2 files."""
    out = []
    codes = ISO[gas]
    with open(path, "rb") as f:
        for l in f:
            o = offset
            v = float(l[3 + o:15 + o])
            if v < lo:
                continue
            if v > hi:
                break
            s = float(l[15 + o:25 + o])
            if s < smin:
                continue
            c = l[2 + o:3 + o]
            if c not in [codes[k:k + 1] for k in range(len(codes))]:
                continue
            iso = codes.index(c)
            if offset == 0:
                ga, gs, e, n = float(l[35:40]), float(l[40:45]), float(l[45:55]), float(l[55:59])
            else:
                ga, gs, e, n = float(l[38:44]), float(l[44:49]), float(l[49:59]), float(l[59:65])
            out.append((v, s, ga, gs, max(e, 0.0), n, iso))
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

def k_lines(lines, gas, t, p_atm, gidx, gscale, cut, chi, mass):
    """Absorption cross-section (cm2/molecule) on the flux grid."""
    v, s, e, n, iso = lines[:, 0], lines[:, 1], lines[:, 4], lines[:, 5], lines[:, 6].astype(int)
    st = (s * qratio(gas, iso, t) * np.exp(-C2 * e * (1 / t - 1 / 296.0))
          * (1 - np.exp(-C2 * v / t)) / (1 - np.exp(-C2 * v / 296.0)))
    gam = lines[:, gidx] * gscale * p_atm * (296.0 / t) ** n
    gd = v * np.sqrt(2 * KB * t * np.log(2) / (mass * AMU)) / 2.99792458e8
    gam = np.maximum(np.maximum(gam, gd), DNU)
    lg = np.log10(gam)
    cls = np.floor((lg - lg.min()) / 0.02).astype(int)
    x = (v - GLO) / DNU
    i0 = np.floor(x).astype(int); fr = x - i0
    nwin = int(round(cut / DNU))
    d = np.arange(-nwin, nwin + 1) * DNU
    k = np.zeros_like(grid)
    for c in np.unique(cls):
        m = cls == c
        dep = (np.bincount(np.clip(i0[m], 0, grid.size - 1), st[m] * (1 - fr[m]), grid.size)
               + np.bincount(np.clip(i0[m] + 1, 0, grid.size - 1), st[m] * fr[m], grid.size))
        g = 10 ** (lg.min() + 0.02 * (c + 0.5))
        ker = g / np.pi / (d * d + g * g)
        if chi:
            ker = ker * p_h_chi(d, t)
        k += fftconvolve(dep, ker, mode="same")
    return np.maximum(k[inr], 0.0)

def read_cia(path):
    blocks, lines = [], open(path).readlines()
    i = 0
    while i < len(lines):
        h = lines[i].split(); npt = int(h[3])
        dd = np.array([[float(x) for x in lines[i + 1 + j].split()[:2]] for j in range(npt)])
        y = np.interp(nu, dd[:, 0], np.maximum(dd[:, 1], 0.0), left=np.nan, right=np.nan)
        blocks.append((float(h[4]), y)); i += npt + 1
    return np.array([b[0] for b in blocks]), np.array([b[1] for b in blocks])
CIA = read_cia(cia_f)
def k_cia(t):
    temps, vals = CIA
    cover = ~np.isnan(vals)
    tt = temps[:, None] * np.ones_like(vals)
    below = np.where(cover & (tt <= t), tt, -np.inf)
    above = np.where(cover & (tt >= t), tt, np.inf)
    ib, ia = below.argmax(axis=0), above.argmin(axis=0)
    tb, ta = below.max(axis=0), above.min(axis=0)
    col = np.arange(nu.size)
    hb, ha = np.isfinite(tb), np.isfinite(ta)
    ib = np.where(hb, ib, ia); ia = np.where(ha, ia, ib)
    tb = np.where(hb, tb, ta); ta = np.where(ha, ta, tb)
    with np.errstate(invalid="ignore"):
        w = np.where(ta > tb, (t - tb) / np.where(ta > tb, ta - tb, 1.0), 0.0)
    k = (1 - w) * vals[ib, col] + w * vals[ia, col]
    return np.where(cover.any(axis=0), k, 0.0)

co2_hitran = read_lines(f"{hdir}/HITRAN2020_CO2.par", "CO2", 0, 0.0, GLO, GHI)
co2_hitemp = read_lines(hitemp_f, "CO2", 0, 0.0, GLO, GHI)
h2o = read_lines(f"{hdir}/HITRAN2020_H2O.par", "H2O", 0, 0.0, LO - 25, HI + 25)
so2 = read_lines(f"{hdir}/HITRAN2020_SO2.par_co2", "SO2", 3, 1e-26, LO - 25, HI + 25)
print(f"lines {GLO:.0f}-{GHI:.0f} cm-1: CO2 HITRAN {len(co2_hitran)}, CO2 HITEMP filtered "
      f"{len(co2_hitemp)}; H2O {len(h2o)}, SO2 {len(so2)} ({LO-25:.0f}-{HI+25:.0f})")
print(f"{len(layers)} layers, {p_lev[0]:.0f} Pa to {p_lev[-1]:.3g} Pa, T {t_lev[0]:.0f}-{T_SURF:.0f} K",
      flush=True)

for L in layers:
    pa, t = L["p"] / 101325, L["t"]
    nco2 = L["xco2"] * L["p"] / (KB * t) * 1e-6
    L["cia"] = k_cia(t) * nco2 ** 2 * L["dz"]
    L["h2o"] = k_lines(h2o, "H2O", t, pa, 2, 1.5, 25.0, False, 18.0) * L["ncol"] * L["xh2o"]
    L["so2"] = k_lines(so2, "SO2", t, pa, 2, 1.0, 25.0, False, 64.0) * L["ncol"] * L["xso2"]
    L["hitran"] = k_lines(co2_hitran, "CO2", t, pa, 3, 1.0, CUT, True, 44.0) * L["ncol"] * L["xco2"]
    L["hitemp"] = k_lines(co2_hitemp, "CO2", t, pa, 3, 1.0, CUT, True, 44.0) * L["ncol"] * L["xco2"]

def planck(t):
    return C1 * nu ** 3 / (np.exp(C2 * nu / t) - 1)

MU = 1 / 1.66
def run(names, sel=slice(None)):
    """Up/down fluxes at levels. The Planck source varies linearly in optical depth across each
    layer between the level temperatures, so optically thick layers carry the diffusion flux
    (~ dB/dtau) rather than a jump between layer-mean Planck functions."""
    nl = len(layers)
    pl = lambda t: np.pi * planck(t)[sel]
    up = [None] * (nl + 1); dn = [None] * (nl + 1)
    up[nl] = pl(T_SURF)
    tau, tr, g = [], [], []
    for L in layers:
        tu = np.maximum(sum(L[n] for n in names)[sel] / MU, 1e-10)
        e = np.exp(-tu)
        tau.append(tu); tr.append(e)
        # [1 - (1 + tau) e^-tau] / tau, with its small-tau limit tau/2
        g.append(np.where(tu > 1e-4, (1 - (1 + tu) * e) / tu, tu / 2))
    bl = [pl(t) for t in t_lev]
    for i in range(nl - 1, -1, -1):
        bt, bb = bl[i], bl[i + 1]
        up[i] = up[i + 1] * tr[i] + bt * (1 - tr[i]) + (bb - bt) * g[i]
    dn[0] = np.zeros_like(up[nl])
    for i in range(nl):
        bt, bb = bl[i], bl[i + 1]
        dn[i + 1] = dn[i] * tr[i] + bb * (1 - tr[i]) - (bb - bt) * g[i]
    net = np.array([(u - d).sum() * DNU for u, d in zip(up, dn)])
    return net, up[0].sum() * DNU

base = ["cia", "h2o", "so2"]
cases = {"CIA+H2O+SO2": base, "+ HITRAN CO2": base + ["hitran"], "+ HITEMP CO2 (filtered)": base + ["hitemp"]}
res = {k: run(v) for k, v in cases.items()}
print(f"\nOLR {LO:.0f}-{HI:.0f} cm-1 (W m-2): " + ", ".join(f"{k}: {v[1]:.4f}" for k, v in res.items()))

print("\nLevel   p (bar)    T (K)   net flux (W m-2) [no CO2 lines | HITRAN | HITEMP]  HITEMP/HITRAN-1"
      "   heating (K/day) HITRAN | HITEMP | diff")
n0, n1, n2 = (res[k][0] for k in cases)
for i in range(0, len(p_lev) - 1, 2):
    dp = p_lev[i + 1] - p_lev[i]
    h1 = (n1[i + 1] - n1[i]) / dp * G / CP * 86400
    h2 = (n2[i + 1] - n2[i]) / dp * G / CP * 86400
    rel = n2[i] / n1[i] - 1 if n1[i] != 0 else 0.0
    print(f"{i:3d}  {p_lev[i]/1e5:9.4f}  {t_lev[i]:6.0f}   {n0[i]:9.4f} {n1[i]:9.4f} {n2[i]:9.4f}   "
          f"{rel:+8.2%}       {h1:9.4f} {h2:9.4f} {h2-h1:+9.4f}")

print("\nPer 100 cm-1 bin: OLR and net flux at the level nearest 30 bar, HITRAN -> HITEMP")
i30 = int(np.argmin(np.abs(p_lev - 3e6)))
for b0 in (1800, 1900, 2000):
    m = (nu >= b0) & (nu < b0 + 100)
    r1, r2 = run(base + ["hitran"], m), run(base + ["hitemp"], m)
    print(f"  {b0}-{b0+100}: OLR {r1[1]:.4f} -> {r2[1]:.4f} ({r2[1]/r1[1]-1:+.2%}); "
          f"net at {p_lev[i30]/1e5:.1f} bar {r1[0][i30]:.4f} -> {r2[0][i30]:.4f} ({r2[0][i30]/r1[0][i30]-1:+.2%})")

for lab, sel in [("upper (p < 1 bar)", lambda L: L["p"] < 1e5),
                 ("middle (1-10 bar)", lambda L: 1e5 <= L["p"] <= 1e6),
                 ("deep (p > 10 bar)", lambda L: L["p"] > 1e6)]:
    tsum = {n: sum(L[n] for L in layers if sel(L)) for n in ["cia", "h2o", "so2", "hitran", "hitemp"]}
    print(f"\n{lab}: column tau over {LO:.0f}-{HI:.0f} cm-1, median (10th percentile): "
          + ", ".join(f"{n} {np.median(v):.3g} ({np.percentile(v, 10):.3g})" for n, v in tsum.items()))
    for b0 in (1800, 1900, 2000):
        m = (nu >= b0) & (nu < b0 + 100)
        print(f"   {b0}-{b0+100}: " + ", ".join(f"{n} {np.median(v[m]):.3g} ({np.percentile(v[m], 10):.3g})"
                                        for n, v in tsum.items()))
