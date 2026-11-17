#!/usr/bin/env python3
"""Fortran tables for the Haus-heating idealised Venus forcing.

Reads SolarNetFlux_RH.dat, the net solar flux table of Haus et al. that the
LMD Venus PCM reads (sw_venus_rh.F), and writes the parameter arrays of
venus_heating_forcings_mod.F90 on stdout:

  log10_p_tab(nlay)            log10 of layer pressure (Pa), top first
  lat_tab(nlat)                table latitudes (degrees)
  q_tab(nlay, nhour, nlat)     specific solar heating g dF/dp (W kg-1)
  q_mean_tab(nlay)             its global and 24-hour mean (W kg-1)
  tau_tab(nlay)                radiative relaxation time (s)

The file holds, per latitude (0-85 every 5 degrees, and 89), the net downward
flux at 119 levels from 0 to 140 km for hour angles 0, 15, ..., 90 degrees
from noon (zero obliquity; the zenith angle is capped at 89 degrees). Its
fluxes are half the physical values; with the factor 2 applied, as in the PCM,
the global-mean flux absorbed by the planet is 158 W m-2, matching a Bond
albedo of 0.76.

Layers lie between table levels up to 100 km. The heating of a layer is
g (F_top - F_bot) / (p_bot - p_top), placed at the log-pressure midpoint. The
flux reaching the surface is spread uniformly by mass over the layers below
10 km, keeping its latitude and hour dependence, since the model has no
surface to absorb it.

q_mean_tab is the area and 24-hour mean of q as the model interpolates it:
linear in latitude and in hour angle up to 90 degrees, zero beyond. With it
subtracted, the forcing integrates to about zero over each pressure level.

tau_tab follows the Newtonian-cooling curve of Crisp (1989, Icarus 77,
391-413, Fig. 12) above 60 km, read by eye, and rises log-linearly in time
from its 60 km value to --tau-deep at 45 km and below. Heights are converted
to pressure with the table's own altitude and pressure columns.

Diagnostics go to stderr. With --check, the heating is also integrated over
a 60-level, 90 km column at the levels and pressures of the T_Z profile of
Lee et al. (2007), as a check that the model's pointwise sampling keeps the
column budget.
"""
import argparse
import math
import sys

import numpy as np

# Crisp (1989) Fig. 12, Newtonian cooling: (height km, relaxation time days),
# read by eye from the figure.
CRISP_TAU = [(100.0, 0.3), (95.0, 0.5), (90.0, 1.5), (85.0, 2.5),
             (80.0, 4.0), (75.0, 6.0), (70.0, 9.0), (65.0, 10.0),
             (62.0, 15.0), (60.0, 20.0)]

NHOUR = 7
TRAPZ = getattr(np, "trapezoid", None) or np.trapz
HOUR_DEG = 15.0


def read_table(path):
    """Altitude (km), pressure (Pa) and flux (W m-2) [level, hour, lat]."""
    lines = open(path).read().splitlines()
    nlev = 119
    alt = np.array([float(l.split()[0]) for l in lines[:nlev]])
    pres = np.array([float(l.split()[1]) for l in lines[:nlev]]) * 1.0e5
    lats, fluxes = [], []
    i = nlev
    while i < len(lines):
        if lines[i].startswith("Latitude"):
            lats.append(float(lines[i].split()[1]))
            rows = [[float(x) for x in lines[j].split()[1:1 + NHOUR]]
                    for j in range(i + 4, i + 4 + nlev)]
            fluxes.append(rows)
            i += 4 + nlev
        else:
            i += 1
    flux = 2.0 * np.array(fluxes).transpose(1, 2, 0)
    return alt, pres, np.array(lats), flux


def tau_days(z_km, tau_deep):
    """Relaxation time (days) at height z (km)."""
    zs = [z for z, _ in CRISP_TAU][::-1]
    ts = [t for _, t in CRISP_TAU][::-1]
    if z_km >= 60.0:
        return float(np.exp(np.interp(z_km, zs, np.log(ts))))
    if z_km <= 45.0:
        return tau_deep
    w = (60.0 - z_km) / 15.0
    return float(np.exp((1.0 - w) * np.log(ts[0]) + w * np.log(tau_deep)))


def interp_q(q, lat_tab, lat_deg, hour_deg):
    """q [lay, hour, lat] at a latitude and hour angle, as the model does."""
    h = abs(hour_deg)
    if h > 90.0:
        return np.zeros(q.shape[0])
    a = abs(lat_deg)
    j = min(np.searchsorted(lat_tab, a, side="right") - 1, len(lat_tab) - 2)
    wl = min(max((a - lat_tab[j]) / (lat_tab[j + 1] - lat_tab[j]), 0.0), 1.0)
    k = min(int(h // HOUR_DEG), NHOUR - 2)
    wh = (h - HOUR_DEG * k) / HOUR_DEG
    lo = q[:, k, j] + wh * (q[:, k + 1, j] - q[:, k, j])
    hi = q[:, k, j + 1] + wh * (q[:, k + 1, j + 1] - q[:, k, j + 1])
    return lo + wl * (hi - lo)


def global_mean(q, lat_tab, nlat_fine=361, nhour_fine=361):
    """Area and 24-hour mean of the interpolated q, per layer."""
    lats = np.linspace(0.0, 90.0, nlat_fine)
    hours = np.linspace(0.0, 180.0, nhour_fine)
    wlat = np.cos(np.radians(lats))
    total = np.zeros(q.shape[0])
    for lat, w in zip(lats, wlat):
        col = TRAPZ([interp_q(q, lat_tab, lat, h) for h in hours],
                       hours, axis=0) / 180.0
        total += w * col
    return total / wlat.sum()


def fortran_array(name, values, per_line=4):
    """Fortran continuation lines for an r_def parameter array."""
    items = [f"{v: .6E}_r_def" for v in values]
    lines = [", ".join(items[i:i + per_line])
             for i in range(0, len(items), per_line)]
    body = ",  &\n       ".join(lines)
    return f"(/  &\n       {body}  /)"


def check_column(log10_p, q_mean, q, lat_tab, p_surf, gravity, rd):
    """Column-integrated heating on a 60-level 90 km grid vs the table."""
    t_z = (733.4861, 301.2670, 47.2002, -5.0230, -2.3257, -0.1896)
    z = np.linspace(0.0, 9.0e4, 61)
    lnp = [math.log(p_surf)]
    for i in range(60):
        eta = math.exp(lnp[-1]) / p_surf
        x = math.log10(max(eta, 2.265e-6))
        t = sum(c * x**n for n, c in enumerate(t_z))
        lnp.append(lnp[-1] - gravity * (z[i + 1] - z[i]) / (rd * t))
    p = np.exp(lnp)
    xp = log10_p
    # Wtheta points at level boundaries; each carries the mass between the
    # midpoints of its neighbouring layers.
    mass = np.empty(61)
    mass[0] = (p[0] - 0.5 * (p[0] + p[1])) / gravity
    mass[-1] = (0.5 * (p[-2] + p[-1])) / gravity
    mass[1:-1] = 0.5 * (p[:-2] - p[2:]) / gravity
    def sample(prof):
        return np.interp(np.log10(p), xp, prof)
    absorbed_mean = float(np.sum(mass * sample(q_mean)))
    noon_eq = float(np.sum(mass * sample(interp_q(q, lat_tab, 0.0, 0.0))))
    return absorbed_mean, noon_eq


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("table", help="SolarNetFlux_RH.dat")
    ap.add_argument("--top", type=float, default=100.0,
                    help="highest table level used, km (default: 100)")
    ap.add_argument("--surface-depth", type=float, default=10.0,
                    help="depth over which the surface flux is spread, km "
                         "(default: 10)")
    ap.add_argument("--tau-deep", type=float, default=2000.0,
                    help="relaxation time at 45 km and below, Earth days "
                         "(default: 2000)")
    ap.add_argument("--gravity", type=float, default=8.87,
                    help="gravitational acceleration, m/s^2 (default: 8.87)")
    ap.add_argument("--check", action="store_true",
                    help="integrate over a 60-level model column too")
    args = ap.parse_args()

    alt, pres, lat_tab, flux = read_table(args.table)
    keep = alt <= args.top + 1e-6
    alt, pres, flux = alt[keep], pres[keep], flux[keep]
    if np.any(np.diff(alt) >= 0):
        sys.exit("table levels are not in decreasing altitude order")

    g = args.gravity
    dp = pres[1:] - pres[:-1]
    q = g * (flux[:-1] - flux[1:]) / dp[:, None, None]
    z_mid = 0.5 * (alt[:-1] + alt[1:])
    log10_p = 0.5 * (np.log10(pres[:-1]) + np.log10(pres[1:]))

    i_base = int(np.argmin(np.abs(alt - args.surface_depth)))
    if abs(alt[i_base] - args.surface_depth) > 1e-6:
        sys.exit(f"no table level at {args.surface_depth} km")
    deep = alt[1:] < args.surface_depth + 1e-6
    q[deep] += g * flux[-1][None, :, :] / (pres[-1] - pres[i_base])

    q_mean = global_mean(q, lat_tab)
    tau = np.array([tau_days(z, args.tau_deep) for z in z_mid]) * 86400.0

    nlay = q.shape[0]
    absorbed = float(np.sum(q_mean * dp) / g)
    report = [f"{nlay} layers, {alt[0]:.0f} to {alt[-1]:.0f} km, "
              f"{len(lat_tab)} latitudes, {NHOUR} hour angles",
              f"global-mean absorbed in the column from the table: "
              f"{absorbed:.1f} W m-2 (TOA net flux {flux[0].mean():.1f} "
              f"unweighted)",
              f"tau {tau.min() / 86400:.2f} to {tau.max() / 86400:.0f} days"]
    if args.check:
        col, noon = check_column(log10_p, q_mean, q, lat_tab, pres[-1], g,
                                 191.4)
        report.append(f"60-level column, sampled at Wtheta points: "
                      f"global mean {col:.1f} W m-2, equator noon "
                      f"{noon:.1f} W m-2")
    print("\n".join(report), file=sys.stderr)

    q_flat = q.transpose(0, 1, 2).reshape(-1, order="F")
    out = [
        f"  !> Number of layers in the tables.",
        f"  integer(kind=i_def), parameter :: nlay = {nlay}",
        f"  !> Number of hour angles in q_tab: 0 to 90 degrees from noon.",
        f"  integer(kind=i_def), parameter :: nhour = {NHOUR}",
        f"  !> Number of latitudes in q_tab.",
        f"  integer(kind=i_def), parameter :: nlat = {len(lat_tab)}",
        "",
        "  !> log10 of layer pressure (Pa), ascending.",
        "  real(kind=r_def), parameter :: log10_p_tab(nlay) = "
        + fortran_array("log10_p_tab", log10_p),
        "",
        "  !> Table latitudes (degrees).",
        "  real(kind=r_def), parameter :: lat_tab(nlat) = "
        + fortran_array("lat_tab", lat_tab),
        "",
        "  !> Specific solar heating (W kg-1) by layer, hour angle and latitude.",
        "  real(kind=r_def), parameter :: q_tab(nlay,nhour,nlat) = reshape( "
        + fortran_array("q_tab", q_flat) + ", (/ nlay, nhour, nlat /) )",
        "",
        "  !> Global and 24-hour mean of q_tab (W kg-1) by layer.",
        "  real(kind=r_def), parameter :: q_mean_tab(nlay) = "
        + fortran_array("q_mean_tab", q_mean),
        "",
        "  !> Radiative relaxation time (s) by layer.",
        "  real(kind=r_def), parameter :: tau_tab(nlay) = "
        + fortran_array("tau_tab", tau),
    ]
    print("\n".join(out))


if __name__ == "__main__":
    main()
