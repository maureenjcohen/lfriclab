"""Extra far-wing absorption of Bezard (2011) chi over Pollack et al. (1993) chi in the 1.74 and
2.3 um windows, as an equivalent CO2-CO2 continuum (cm-1 amagat-2). HITRAN2020 CO2, self widths.
Usage: python3 -I chi_continuum.py <HITRAN2020_CO2.par> <parsum.dat>"""
import sys, numpy as np
par, parsum = sys.argv[1:3]
C2, AMAGAT = 1.4387769, 2.6868e19
hdr = open(parsum).readline().split(); j = hdr.index("CO2_7")
T_, Q_ = [], []
for l in open(parsum).readlines()[1:]:
    w = l.split()
    if float(w[0]) > 1000: break
    T_.append(float(w[0])); Q_.append(float(w[j]))
q = lambda t: np.interp(t, T_, Q_)
def chi_pollack(d):
    d = np.abs(d); c = np.zeros_like(d)
    c[d < 3] = 1; m = (d >= 3) & (d < 10); c[m] = 1.35 * np.exp(-d[m] / 10)
    m = (d >= 10) & (d < 120); c[m] = 0.614 * np.exp(-d[m] / 47)
    return c
def chi_bezard(d):
    d = np.abs(d); c = np.zeros_like(d)
    c[d < 3] = 1; m = (d >= 3) & (d < 20); c[m] = 1.2214 * np.exp(-d[m] / 15)
    m = (d >= 20) & (d < 120); c[m] = 0.3477 * np.exp(-d[m] / 260)
    m = (d >= 120) & (d < 250); c[m] = 0.7276 * np.exp(-d[m] / 100)
    return c
def bez_norm(g):
    """Bezard renormalises the chi-weighted Lorentzian to unit area within the cut-off."""
    d = np.arange(-250, 250.0001, 0.01)
    return 1.0 / np.trapezoid(g / np.pi / (d * d + g * g) * chi_bezard(d), d)
cases = [("2.3 um window", 4150, 4550, 10.0, 500.0), ("1.74 um window", 5650, 5850, 20.0, 575.0)]
for lab, lo, hi, p, t in cases:
    nu = np.arange(lo, hi, 0.05)
    lines = []
    for l in open(par):
        v = float(l[3:15])
        if lo - 250 <= v <= hi + 250:
            lines.append((v, float(l[15:25]), float(l[40:45]), float(l[45:55]), float(l[55:59])))
    L = np.array(lines); v, s, gs, e, n = L.T
    st = s * q(296) / q(t) * np.exp(-C2 * e * (1 / t - 1 / 296)) * (1 - np.exp(-C2 * v / t)) / (1 - np.exp(-C2 * v / 296))
    gam = gs * p * (296 / t) ** n
    kp = np.zeros_like(nu); kb = np.zeros_like(nu)
    nb = bez_norm(np.median(gam))
    for vi, si, gi in zip(v, st, gam):
        d = nu - vi
        far = np.abs(d) < 250
        if not far.any(): continue
        dd = d[far]; lor = gi / np.pi / (dd * dd + gi * gi)
        kp[far] += si * lor * chi_pollack(dd)
        kb[far] += si * lor * chi_bezard(dd) * nb
    rho = p * 273.15 / t / 1.01325          # amagat of CO2 (taken as pure CO2)
    dk = kb - kp                             # cm2/molecule
    keq = dk * AMAGAT / rho                  # cm-1 amagat-2
    print(f"{lab}: {len(L)} lines, P={p} bar, T={t} K, rho={rho:.2f} amagat, Bezard norm {nb:.3f}")
    print(f"   mean sigma Pollack {kp.mean():.2e}, Bezard {kb.mean():.2e} cm2/molec;"
          f" extra wing as continuum: mean {keq.mean():.2e}, median {np.median(keq):.2e},"
          f" 10th pct {np.percentile(keq,10):.2e} cm-1 amagat-2")
