#!/usr/bin/env python3
"""Potential temperature table for an atmosphere of constant buoyancy frequency
under c_p(T).

The true buoyancy frequency is

  N^2 = (g/T) (dT/dz + g/c_p(T))

so holding N fixed gives dT/dz = N^2 T/g - g/c_p(T). With hydrostatic balance,
dp/dz = -g p / (R T), this gives T(z) and p(z). The table holds the model's
theta = T (p0/p)^(R/cp), built on the namelist cp. A model that uses cp in
place of c_p(T) sees the buoyancy frequency

  N_cp^2 = g dln(theta)/dz = N^2 + (g^2/T) (1/cp - 1/c_p(T))

which is printed alongside N, level by level, on stderr.

With --floor, N is instead the smaller of the two at every height: the true
N^2 is raised wherever c_p(T) < cp, so that N_cp^2 stays at N^2. A deep
profile then stays stable under both laws where it crosses c_p(T) = cp.

Gravity is constant, as in planar geometry. The output is a Rose
[namelist:initial_temperature] fragment of profile_data, profile_heights and
profile_size, in the layout of rose-app-venus.conf.
"""
import argparse
import sys

import numpy as np
from scipy.integrate import solve_ivp

from variable_cp_diagnostics import cp_law


def true_n_squared(t, n_bvf, gravity, cp, law, floor):
    """True N^2 (1/s^2) at temperature t; with floor, raised so N_cp >= n_bvf."""
    if not floor:
        return n_bvf**2
    return n_bvf**2 + max(0.0, gravity**2 / t
                          * (1.0 / cp_law(t, *law) - 1.0 / cp))


def lapse(t, n_bvf, gravity, cp, law, floor, t_iso):
    """dT/dz (K/m) at temperature t; zero once t has fallen to t_iso."""
    n2 = true_n_squared(t, n_bvf, gravity, cp, law, floor)
    dtdz = n2 * t / gravity - gravity / cp_law(t, *law)
    if t <= t_iso:
        dtdz = max(dtdz, 0.0)
    return dtdz


def stratified_profile(heights, t_surf, p_surf, n_bvf, gravity, rd, law,
                       cp=900.0, floor=False, t_iso=0.0):
    """Temperature (K) and pressure (Pa) at heights (m)."""
    def rhs(z, y):
        t, lnp = y
        return [lapse(t, n_bvf, gravity, cp, law, floor, t_iso),
                -gravity / (rd * t)]

    sol = solve_ivp(rhs, (heights[0], heights[-1]), [t_surf, np.log(p_surf)],
                    t_eval=heights, rtol=1e-12, atol=1e-12)
    return sol.y[0], np.exp(sol.y[1])


def rose_list(name, values, per_line=6):
    """A Rose list setting, with '=' continuation lines."""
    items = [f"{v:.6E}" for v in values]
    lines = [",".join(items[i:i + per_line])
             for i in range(0, len(items), per_line)]
    pad = " " * len(name)
    out = [f"{name}={lines[0]}"]
    out += [f"{pad}={line}" for line in lines[1:]]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--t-surf", type=float, required=True,
                    help="temperature at the bottom, K")
    ap.add_argument("--p-surf", type=float, required=True,
                    help="pressure at the bottom, Pa")
    ap.add_argument("--n-bvf", type=float, required=True,
                    help="true buoyancy frequency under c_p(T), 1/s")
    ap.add_argument("--floor", action="store_true",
                    help="treat --n-bvf as a floor on both the true N and "
                         "N_cp")
    ap.add_argument("--t-iso", type=float, default=0.0,
                    help="temperature at and below which the profile is "
                         "isothermal, K (default: 0, none)")
    ap.add_argument("--depth", type=float, default=1.0e4,
                    help="height of the top of the table, m (default: 1e4)")
    ap.add_argument("--points", type=int, default=100,
                    help="number of points in the table, at most 100 "
                         "(default: 100)")
    ap.add_argument("--gravity", type=float, default=8.87,
                    help="gravitational acceleration, m/s^2 (default: 8.87)")
    ap.add_argument("--rd", type=float, default=191.4,
                    help="specific gas constant, J/kg/K (default: 191.4)")
    ap.add_argument("--cp", type=float, default=900.0,
                    help="namelist cp on which theta is built, J/kg/K "
                         "(default: 900)")
    ap.add_argument("--p-zero", type=float, default=9.2e6,
                    help="reference pressure, Pa (default: 9.2e6)")
    ap.add_argument("--cp-law-ref", type=float, default=1000.0,
                    help="heat capacity at the law's reference temperature, "
                         "J/kg/K (default: 1000)")
    ap.add_argument("--cp-law-t0", type=float, default=460.0,
                    help="reference temperature of the law, K (default: 460)")
    ap.add_argument("--cp-law-exponent", type=float, default=0.35,
                    help="exponent of the law (default: 0.35)")
    args = ap.parse_args()

    # profile_interp_kernel_mod holds the table in arrays of 100 elements and
    # does not check profile_size, so a longer table overruns them.
    n = args.points
    if not 2 <= n <= 100:
        sys.exit(f"{n} points; the table must have 2 to 100 points")
    heights = np.linspace(0.0, args.depth, n)
    law = (args.cp_law_ref, args.cp_law_t0, args.cp_law_exponent)

    temperature, pressure = stratified_profile(
        heights, args.t_surf, args.p_surf, args.n_bvf, args.gravity, args.rd,
        law, args.cp, args.floor, args.t_iso)
    theta = temperature * (args.p_zero / pressure) ** (args.rd / args.cp)

    cp = cp_law(temperature, *law)
    dtdz = np.array([lapse(t, args.n_bvf, args.gravity, args.cp, law,
                           args.floor, args.t_iso) for t in temperature])
    n_true = np.sqrt(args.gravity / temperature
                     * (dtdz + args.gravity / cp))
    n_cp = np.sqrt(n_true**2 + args.gravity**2 / temperature
                   * (1.0 / args.cp - 1.0 / cp))
    # Check against the table itself: g dln(theta)/dz at mid-points.
    z_mid = 0.5 * (heights[1:] + heights[:-1])
    n_cp_table = np.sqrt(args.gravity * np.diff(np.log(theta))
                         / np.diff(heights))
    n_cp_mid = np.interp(z_mid, heights, n_cp)
    print(f"T {temperature[0]:.2f} -> {temperature[-1]:.2f} K, "
          f"p {pressure[0]:.4E} -> {pressure[-1]:.4E} Pa, "
          f"c_p {cp[0]:.1f} -> {cp[-1]:.1f} J/kg/K, "
          f"theta {theta[0]:.2f} -> {theta[-1]:.2f} K\n"
          f"N {n_true.min():.4E} -> {n_true.max():.4E} 1/s; "
          f"N_cp {n_cp.min():.4E} -> {n_cp.max():.4E} 1/s "
          f"(period {2 * np.pi / n_cp.max():.0f} -> "
          f"{2 * np.pi / n_cp.min():.0f} s)\n"
          f"N_cp from the table: max relative difference "
          f"{np.max(np.abs(n_cp_table / n_cp_mid - 1.0)):.1E}",
          file=sys.stderr)

    print("[namelist:initial_temperature]")
    print(rose_list("profile_data", theta))
    print(rose_list("profile_heights", heights))
    print(f"profile_size={n}")


if __name__ == "__main__":
    main()
