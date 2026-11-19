"""Overlap screen: H2SO4 and SO3 vapour against CO2/SO2/H2O lines and CO2-CO2 CIA
in the sub-cloud layer (1.05-3.97 bar), monochromatic line-by-line.
Usage: python3 -I h2so4_so3_overlap_screen.py <hitran_dir> <parsum.dat> <cia file> <h2so4 txt> <so3 txt>
"""
import sys
import numpy as np

hdir, parsum_f, cia_f, h2so4_f, so3_f = sys.argv[1:6]

C2 = 1.4387769          # cm K
KB = 1.380649e-23
AMU = 1.66053907e-27
G = 8.87
M_AIR = 43.4
NU0, NU1, DNU = 500.0, 2300.0, 0.02
nu = np.arange(NU0, NU1 + DNU / 2, DNU)
LINE_LO, LINE_HI = NU0 - 100.0, NU1 + 100.0

# Profile levels from venus_atm_obs_so2.raw (Pa, K); mass ratios converted to vmr.
p_lev = np.array([1.05e5, 1.46e5, 2.04e5, 2.84e5, 3.97e5])
t_lev = np.array([341.0, 362.0, 380.0, 400.0, 424.0])
x_co2 = 0.977 * M_AIR / 44.01
x_so2 = 1.5e-4 * M_AIR / 64.07
x_h2o = 1.14e-5 * M_AIR / 18.02
layers = []
for i in range(4):
    pt, pb = p_lev[i], p_lev[i + 1]
    p = np.sqrt(pt * pb)
    t = np.interp(np.log(p), np.log(p_lev), t_lev)
    ncol = (pb - pt) / (M_AIR * AMU * G) * 1e-4          # molecules cm-2
    dz = KB * t / (M_AIR * AMU * G) * np.log(pb / pt) * 100.0  # cm
    layers.append(dict(p=p, t=t, ncol=ncol, dz=dz))

# Partition sums
hdr = open(parsum_f).readline().split()
_cols = [hdr.index(c) for c in ("Temp(K)", "H2O_1", "CO2_7", "SO2_42")]
_rows = []
with open(parsum_f) as f:
    next(f)
    for l in f:
        w = l.split()
        if float(w[0]) > 1000:
            break
        _rows.append([float(w[j]) for j in _cols])
ps = np.array(_rows)
_qidx = {"H2O_1": 1, "CO2_7": 2, "SO2_42": 3}
def qratio(col, t):
    j = _qidx[col]
    return np.interp(296.0, ps[:, 0], ps[:, j]) / np.interp(t, ps[:, 0], ps[:, j])

def read_lines(path, offset, smin):
    """Fixed-width HITRAN records; offset = 0 for .par, 3 for .par_co2."""
    out = []
    with open(path) as f:
        for l in f:
            o = offset
            v = float(l[3 + o:15 + o])
            if v < LINE_LO or v > LINE_HI:
                continue
            s = float(l[15 + o:25 + o])
            if s < smin:
                continue
            if offset == 0:
                ga, gs = float(l[35:40]), float(l[40:45])
                e, n = float(l[45:55]), float(l[55:59])
            else:
                ga, gs = float(l[38:44]), float(l[44:49])
                e, n = float(l[49:59]), float(l[59:65])
            out.append((v, s, ga, gs, e, n))
    return np.array(out)

def p_h_chi(d, t):
    """Perrin & Hartmann (1989) self-broadened chi, as SOCRATES ip_lpc_co2_ph1989."""
    tc = min(800.0, max(190.0, t))
    b1 = 0.0888 - 0.16 * np.exp(-0.0041 * tc)
    b2 = 0.0526 * np.exp(-0.00152 * tc)
    b3 = 0.0232
    d = np.abs(d)
    chi = np.ones_like(d)
    m = (d >= 3) & (d < 30)
    chi[m] = np.exp(-b1 * (d[m] - 3))
    m = (d >= 30) & (d < 120)
    chi[m] = np.exp(-b1 * 27 - b2 * (d[m] - 30))
    m = d >= 120
    chi[m] = np.exp(-b1 * 27 - b2 * 90 - b3 * (d[m] - 120))
    return chi

def k_lines(lines, t, p_atm, qcol, gamma_idx, gamma_scale, cut, chi):
    v, s, e, n = lines[:, 0], lines[:, 1], lines[:, 4], lines[:, 5]
    st = (s * qratio(qcol, t) * np.exp(-C2 * e * (1 / t - 1 / 296.0))
          * (1 - np.exp(-C2 * v / t)) / (1 - np.exp(-C2 * v / 296.0)))
    gam = lines[:, gamma_idx] * gamma_scale * p_atm * (296.0 / t) ** n
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
    blocks = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        h = lines[i].split()
        npt, tk = int(h[3]), float(h[4])
        d = np.array([[float(x) for x in lines[i + 1 + j].split()[:2]] for j in range(npt)])
        blocks.append((float(h[1]), float(h[2]), tk, d))
        i += npt + 1
    return blocks

def k_cia(blocks, t):
    k = np.zeros_like(nu)
    ranges = sorted(set((b[0], b[1]) for b in blocks))
    for lo, hi in ranges:
        bl = sorted([b for b in blocks if (b[0], b[1]) == (lo, hi)], key=lambda b: b[2])
        temps = np.array([b[2] for b in bl])
        tc = min(max(t, temps[0]), temps[-1])
        j = np.searchsorted(temps, tc)
        j = min(max(j, 1), len(temps) - 1)
        w = (tc - temps[j - 1]) / (temps[j] - temps[j - 1])
        y0 = np.interp(nu, bl[j - 1][3][:, 0], bl[j - 1][3][:, 1], left=0, right=0)
        y1 = np.interp(nu, bl[j][3][:, 0], bl[j][3][:, 1], left=0, right=0)
        m = (nu >= lo) & (nu <= hi)
        k[m] += ((1 - w) * y0 + w * y1)[m]
    return k, temps[-1]

def xsec_file(path, t_k, scale):
    d = np.loadtxt(path, skiprows=8)
    o = np.argsort(d[:, 0])
    n1 = 1e-6 * 101325 / (KB * t_k) * 1e-6     # cm-3 at 1 ppm
    sig = np.log(10) * d[o, 1] / 100.0 / n1     # absorbance per ppm*m -> per ppm*cm
    return scale * np.interp(nu, d[o, 0], sig, left=0, right=0)

KMMOL = 1.6605e-19
def gauss_band(nu0, kmmol, eqw):
    fwhm = eqw / 1.0645
    s = fwhm / 2.3548
    return kmmol * KMMOL / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((nu - nu0) / s) ** 2)

# --- background ---
co2 = read_lines(f"{hdir}/HITRAN2020_CO2.par", 0, 1e-30)
so2 = read_lines(f"{hdir}/HITRAN2020_SO2.par_co2", 3, 1e-28)
h2o = read_lines(f"{hdir}/HITRAN2020_H2O.par", 0, 1e-30)
cia = read_cia(cia_f)
print(f"lines kept: CO2 {len(co2)}, SO2 {len(so2)}, H2O {len(h2o)}")
for L in layers:
    p_atm = L["p"] / 101325
    tco2 = k_lines(co2, L["t"], p_atm, "CO2_7", 3, 1.0, 100.0, True) * L["ncol"] * x_co2
    tso2 = k_lines(so2, L["t"], p_atm, "SO2_42", 2, 1.0, 25.0, False) * L["ncol"] * x_so2
    th2o = k_lines(h2o, L["t"], p_atm, "H2O_1", 2, 1.5, 25.0, False) * L["ncol"] * x_h2o
    kc, tmax = k_cia(cia, L["t"])
    nco2 = x_co2 * L["p"] / (KB * L["t"]) * 1e-6
    tcia = kc * nco2 ** 2 * L["dz"]
    L.update(co2=tco2, so2=tso2, h2o=th2o, cia=tcia)
    print(f"layer p={L['p']/1e5:.2f} bar T={L['t']:.0f} K N={L['ncol']:.2e} cm-2 dz={L['dz']/1e5:.2f} km")

# --- H2SO4 and SO3 cross-sections (cm2) ---
# Absolute scale: Majkowski 1978 / Hintze 2003 band strengths.
sig_h2so4_8um = xsec_file(h2so4_f, 573.15, 277.0 / 381.0)
sig_so3 = xsec_file(so3_f, 623.15, 298.0 / 514.0)
def h2so4_sig(eqw):
    s = sig_h2so4_8um.copy()
    for nu0, km in [(550.5, 47.0), (834.1, 59.0), (891.4, 302.0), (1464.7, 241.0)]:
        s += gauss_band(nu0, km, eqw)
    return s

def planck(t):
    return nu ** 3 / (np.exp(C2 * nu / t) - 1)

def slab(names, idx):
    return sum(layers[i][n] for i in idx for n in names)

slabs = {"A 1.05-2.04 bar": [0, 1], "B 2.04-3.97 bar": [2, 3]}
bg_names = ["co2", "so2", "h2o", "cia"]
ranges = [(500, 650, "H2SO4 550 / SO2 518 region"), (750, 1000, "H2SO4 834/891"),
          (1140, 1270, "H2SO4 1157/1220"), (1340, 1480, "SO3 1391, H2SO4 1465"), (1480, 1600, "6.3 um gap edge"), (1600, 2300, "5-6 um gap")]

for sname, idx in slabs.items():
    ncol = sum(layers[i]["ncol"] for i in idx)
    tmean = np.mean([layers[i]["t"] for i in idx])
    print(f"\n===== Slab {sname}, column {ncol:.2e} cm-2, mean T {tmean:.0f} K")
    tau_bg = slab(bg_names, idx)
    parts = {n: slab([n], idx) for n in bg_names}
    B = planck(tmean)
    cases = [("H2SO4 1 ppm", h2so4_sig(27.0) * ncol * 1e-6),
             ("H2SO4 10 ppm", h2so4_sig(27.0) * ncol * 1e-5),
             ("H2SO4 1 ppm, bands 2x wider", h2so4_sig(54.0) * ncol * 1e-6),
             ("SO3 0.5 ppm", sig_so3 * ncol * 5e-7)]
    for lo, hi, lab in ranges:
        m = (nu >= lo) & (nu < hi)
        tr_bg = np.sum(B[m] * np.exp(-tau_bg[m])) / np.sum(B[m])
        med = {n: np.median(parts[n][m]) for n in bg_names}
        print(f"\n  {lo}-{hi} cm-1 ({lab}): background Planck-weighted transmission {tr_bg:.3f}")
        print("    median tau: " + ", ".join(f"{n} {med[n]:.2g}" for n in bg_names))
        for cname, tg in cases:
            tr = np.sum(B[m] * np.exp(-tau_bg[m] - tg[m])) / np.sum(B[m])
            print(f"    + {cname:28s}: median tau_gas {np.median(tg[m]):7.2g}, transmission {tr:.3f}"
                  f" (absorptance {1-tr_bg:.3f} -> {1-tr:.3f})")
    # 10 cm-1 bands (the R3 layout) where H2SO4 1 ppm changes absorptance by > 0.05
    print("\n  10 cm-1 bands, H2SO4 1 ppm: band, T_bg, T_with")
    tg = h2so4_sig(27.0) * ncol * 1e-6
    for b0 in np.arange(500, 1600, 10):
        m = (nu >= b0) & (nu < b0 + 10)
        t0 = np.mean(np.exp(-tau_bg[m]))
        t1 = np.mean(np.exp(-tau_bg[m] - tg[m]))
        if t0 - t1 > 0.05:
            print(f"    {b0:.0f}-{b0+10:.0f}: {t0:.3f} -> {t1:.3f}")
    print("  10 cm-1 bands, SO3 0.5 ppm:")
    tg = sig_so3 * ncol * 5e-7
    for b0 in np.arange(1310, 1430, 10):
        m = (nu >= b0) & (nu < b0 + 10)
        t0 = np.mean(np.exp(-tau_bg[m]))
        t1 = np.mean(np.exp(-tau_bg[m] - tg[m]))
        if t0 - t1 > 0.01:
            print(f"    {b0:.0f}-{b0+10:.0f}: {t0:.3f} -> {t1:.3f}")
print(f"\nCIA 1000-1800 cm-1 data stop at 350 K and are held there above it.")

# --- opaque-slab metric: radiative diffusion, flux proportional to sum (dB/dT)/tau ---
def dbdt(t):
    x = C2 * nu / t
    return nu ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2 / t ** 2

print("\n===== Diffusion metric: D = sum over nu of (dB/dT)/tau, per slab")
for sname, idx in slabs.items():
    ncol = sum(layers[i]["ncol"] for i in idx)
    tmean = np.mean([layers[i]["t"] for i in idx])
    tau_bg = slab(bg_names, idx)
    w = dbdt(tmean)
    print(f"\n  Slab {sname}: fraction of 500-2300 cm-1 with tau_bg < 1: {np.mean(tau_bg < 1):.4f}, < 10: {np.mean(tau_bg < 10):.4f}")
    gases = [("H2SO4 1 ppm", h2so4_sig(27.0) * ncol * 1e-6),
             ("H2SO4 10 ppm", h2so4_sig(27.0) * ncol * 1e-5),
             ("H2SO4 1 ppm, 2x wider", h2so4_sig(54.0) * ncol * 1e-6),
             ("SO3 0.5 ppm", sig_so3 * ncol * 5e-7)]
    for lo, hi, lab in ranges + [(500, 1600, "500-1600"), (500, 2300, "whole screen")]:
        m = (nu >= lo) & (nu < hi)
        d0 = np.sum(w[m] / tau_bg[m])
        txt = ", ".join(f"{g}: {np.sum(w[m] / (tau_bg[m] + tg[m])) / d0 - 1:+.1%}" for g, tg in gases)
        print(f"    {lo}-{hi}: share of total D {d0 / np.sum(w / tau_bg):.3f}; change {txt}")
    print("    10 cm-1 bands with lowest tau_bg (median in band):")
    meds = [(np.median(tau_bg[(nu >= b) & (nu < b + 10)]), b) for b in np.arange(500, 2300, 10)]
    for mv, b in sorted(meds)[:8]:
        print(f"      {b:.0f}-{b+10:.0f}: {mv:.2g}")
