"""Far-IR pilot: do the filtered HITEMP 2024 CO2 lines below 250 cm-1 matter against CO2-CO2 CIA,
H2O and SO2? Monochromatic, 10-250 cm-1, column from ~100 Pa to the surface
(venus_atm_obs_so2.raw), no scattering, diffusivity 1.66, black surface.
Cases: CIA+H2O+SO2 (no CO2 lines), + HITRAN2020 CO2, + filtered HITEMP CO2.
Usage: python3 -I farir_pilot.py <raw profile> <hitran dir> <hitemp co2 0-350 .par> <parsum> <CO2-CO2 cia>
"""
import sys
import numpy as np

raw, hdir, hitemp_f, parsum_f, cia_f = sys.argv[1:6]
C2, KB, AMU, G, M_AIR = 1.4387769, 1.380649e-23, 1.66053907e-27, 8.87, 43.4
C1 = 1.191042e-8                    # W m-2 sr-1 (cm-1)-4
CP = 1000.0                         # J kg-1 K-1, for heating rates only
NU0, NU1, DNU = 10.0, 250.0, 0.02
nu = np.arange(NU0, NU1 + DNU / 2, DNU)

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

hdr = open(parsum_f).readline().split()
cols = ["Temp(K)", "H2O_1", "CO2_7", "SO2_42"]
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
            if v < lo:
                continue
            if v > hi:
                break
            s = float(l[15 + o:25 + o])
            if s < smin:
                continue
            if offset == 0:
                gs, ga, e, n = float(l[40:45]), float(l[35:40]), float(l[45:55]), float(l[55:59])
            else:
                ga, gs, e, n = float(l[38:44]), float(l[44:49]), float(l[49:59]), float(l[59:65])
            out.append((v, s, ga, gs, e, n))
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

def k_lines(lines, t, p_atm, qcol, gidx, gscale, cut, chi, mass):
    v, s, e, n = lines[:, 0], lines[:, 1], lines[:, 4], lines[:, 5]
    st = (s * qratio(qcol, t) * np.exp(-C2 * e * (1 / t - 1 / 296.0))
          * (1 - np.exp(-C2 * v / t)) / (1 - np.exp(-C2 * v / 296.0)))
    gam = lines[:, gidx] * gscale * p_atm * (296.0 / t) ** n
    gd = v * np.sqrt(2 * KB * t * np.log(2) / (mass * AMU)) / 2.99792458e8
    # Unresolved narrow lines keep their area on the grid (weak lines; differences only)
    gam = np.maximum(np.maximum(gam, gd), DNU)
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
    blocks, lines = [], open(path).readlines()
    i = 0
    while i < len(lines):
        h = lines[i].split(); npt = int(h[3])
        d = np.array([[float(x) for x in lines[i + 1 + j].split()[:2]] for j in range(npt)])
        blocks.append((float(h[1]), float(h[2]), float(h[4]), d)); i += npt + 1
    return blocks
cia = sorted([b for b in read_cia(cia_f) if b[1] <= 760], key=lambda b: b[2])
ctemps = np.array([b[2] for b in cia])
def k_cia(t):
    tc = min(max(t, ctemps[0]), ctemps[-1]); j = min(max(np.searchsorted(ctemps, tc), 1), len(ctemps) - 1)
    w = (tc - ctemps[j - 1]) / (ctemps[j] - ctemps[j - 1])
    y0 = np.interp(nu, cia[j - 1][3][:, 0], cia[j - 1][3][:, 1]); y1 = np.interp(nu, cia[j][3][:, 0], cia[j][3][:, 1])
    return (1 - w) * y0 + w * y1

co2_hitran = read_lines(f"{hdir}/HITRAN2020_CO2.par", 0, 0.0, 0, NU1 + 100)
co2_hitemp = read_lines(hitemp_f, 0, 0.0, 0, NU1 + 100)
h2o = read_lines(f"{hdir}/HITRAN2020_H2O.par", 0, 0.0, 0, NU1 + 25)
so2 = read_lines(f"{hdir}/HITRAN2020_SO2.par_co2", 3, 1e-26, 0, NU1 + 25)
print(f"lines: CO2 HITRAN {len(co2_hitran)}, CO2 HITEMP filtered {len(co2_hitemp)}, H2O {len(h2o)}, SO2 {len(so2)}")
print(f"{len(layers)} layers, {p_lev[0]:.0f} Pa to {p_lev[-1]:.3g} Pa, T {t_lev[0]:.0f}-{T_SURF:.0f} K")

for L in layers:
    pa, t = L["p"] / 101325, L["t"]
    nco2 = L["xco2"] * L["p"] / (KB * t) * 1e-6
    L["cia"] = k_cia(t) * nco2 ** 2 * L["dz"]
    L["h2o"] = k_lines(h2o, t, pa, "H2O_1", 2, 1.5, 25.0, False, 18.0) * L["ncol"] * L["xh2o"]
    L["so2"] = k_lines(so2, t, pa, "SO2_42", 2, 1.0, 25.0, False, 64.0) * L["ncol"] * L["xso2"]
    L["hitran"] = k_lines(co2_hitran, t, pa, "CO2_7", 3, 1.0, 100.0, True, 44.0) * L["ncol"] * L["xco2"]
    L["hitemp"] = k_lines(co2_hitemp, t, pa, "CO2_7", 3, 1.0, 100.0, True, 44.0) * L["ncol"] * L["xco2"]

def planck(t):
    return C1 * nu ** 3 / (np.exp(C2 * nu / t) - 1)

MU = 1 / 1.66
def run(names):
    """Up/down fluxes at levels. The Planck source varies linearly in optical depth across each
    layer between the level temperatures, so optically thick layers carry the diffusion flux
    (~ dB/dtau) rather than a jump between layer-mean Planck functions."""
    nl = len(layers)
    up = [None] * (nl + 1); dn = [None] * (nl + 1)
    up[nl] = np.pi * planck(T_SURF)
    tr, g = [], []
    for L in layers:
        tu = np.maximum(sum(L[n] for n in names) / MU, 1e-10)
        e = np.exp(-tu)
        tr.append(e)
        # [1 - (1 + tau) e^-tau] / tau, with its small-tau limit tau/2
        g.append(np.where(tu > 1e-4, (1 - (1 + tu) * e) / tu, tu / 2))
    bl = [np.pi * planck(t) for t in t_lev]
    for i in range(nl - 1, -1, -1):
        up[i] = up[i + 1] * tr[i] + bl[i] * (1 - tr[i]) + (bl[i + 1] - bl[i]) * g[i]
    dn[0] = np.zeros_like(nu)
    for i in range(nl):
        dn[i + 1] = dn[i] * tr[i] + bl[i + 1] * (1 - tr[i]) - (bl[i + 1] - bl[i]) * g[i]
    net = np.array([(u - d).sum() * DNU for u, d in zip(up, dn)])
    return net, up[0].sum() * DNU

base = ["cia", "h2o", "so2"]
cases = {"CIA+H2O+SO2": base, "+ HITRAN CO2": base + ["hitran"], "+ HITEMP CO2 (filtered)": base + ["hitemp"]}
res = {k: run(v) for k, v in cases.items()}
total_olr = sum(np.pi * planck(T_SURF)) * DNU
print(f"\nOLR 10-250 cm-1 (W m-2): " + ", ".join(f"{k}: {v[1]:.3f}" for k, v in res.items()))
print("\nLevel     p (bar)   T (K)   net flux 10-250 (W m-2) [no CO2 lines | HITRAN | HITEMP]   heating (K/day) HITRAN | HITEMP | diff")
n0, n1, n2 = (res[k][0] for k in cases)
for i in range(0, len(p_lev) - 1, 3):
    dp = p_lev[i + 1] - p_lev[i]
    h1 = -(n1[i + 1] - n1[i]) / dp * G / CP * 86400   # heating = -dFnet/dz / (rho cp) = g/cp dFnet/dp... sign: F_net up
    h2 = -(n2[i + 1] - n2[i]) / dp * G / CP * 86400
    h1, h2 = -h1, -h2
    print(f"{i:3d}  {p_lev[i]/1e5:9.4f}  {t_lev[i]:6.0f}   {n0[i]:9.3f} {n1[i]:9.3f} {n2[i]:9.3f}        {h1:9.4f} {h2:9.4f} {h2-h1:+9.4f}")
# column optical depth comparison in deep and upper parts
for lab, sel in [("upper (p < 1 bar)", lambda L: L["p"] < 1e5), ("deep (p > 10 bar)", lambda L: L["p"] > 1e6)]:
    tsum = {n: sum(L[n] for L in layers if sel(L)) for n in ["cia", "h2o", "so2", "hitran", "hitemp"]}
    print(f"\n{lab}: median column tau over 10-250 cm-1: " + ", ".join(f"{n} {np.median(v):.3g}" for n, v in tsum.items()))
    for b0, b1 in [(10, 50), (50, 100), (100, 150), (150, 200), (200, 250)]:
        m = (nu >= b0) & (nu < b1)
        print(f"   {b0}-{b1}: " + ", ".join(f"{n} {np.median(v[m]):.3g}" for n, v in tsum.items()))
