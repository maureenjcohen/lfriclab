#!/usr/bin/env python3
"""Initial potential temperature table for the Lee et al. (2007) Venus forcing.

Lee, Lewis and Read (2007, J. Geophys. Res., 112, E04S11), eq. 5, relax
temperature towards

  T_eq(eta, lat) = T_Z(eta) + T_H(eta) cos(lat),   eta = p / p_surf

with T_Z a fifth-order polynomial in log10(eta) and T_H tabulated on Lee's 31
levels (Table 1). The coefficient signs are lost in the paper's text; the ones
here reproduce the 31-level T_Z table in the LMD Venus PCM (radlwsw_NewtonCool)
to within 1E-3 K. Lee starts from rest with T = T_Z everywhere, as does this
table.

With hydrostatic balance, dp/dz = -g p / (R T), T_Z gives T(z) and p(z). The
table holds the model's theta = T (p0/p)^(R/cp), built on the namelist cp.

The lapse rate of T_eq at the equator and pole is checked against g/cp, and
printed on stderr, since a superadiabatic target would drive convection that
never stops. Gravity is constant. The output is a Rose
[namelist:initial_temperature] fragment of profile_data, profile_heights and
profile_size.
"""
import argparse
import sys

import numpy as np
from scipy.integrate import solve_ivp

T_Z_COEFFS = (733.4861, 301.2670, 47.2002, -5.0230, -2.3257, -0.1896)

LEE_ETA = np.array([
    9.602e-1, 8.679e-1, 7.577e-1, 6.420e-1, 5.299e-1, 4.273e-1, 3.373e-1,
    2.610e-1, 1.979e-1, 1.472e-1, 1.074e-1, 7.672e-2, 5.361e-2, 3.657e-2,
    2.430e-2, 1.569e-2, 9.814e-3, 5.929e-3, 3.454e-3, 1.934e-3, 1.043e-3,
    5.400e-4, 2.710e-4, 1.324e-4, 6.355e-5, 3.070e-5, 1.525e-5, 7.950e-6,
    4.500e-6, 2.925e-6, 2.265e-6])
LEE_T_H = np.array([
    6.101, 6.136, 6.176, 6.410, 6.634, 6.678, 6.719, 6.762, 7.167, 7.524,
    9.840, 14.948, 21.370, 28.746, 36.373, 43.315, 48.534, 51.175, 50.757,
    47.342, 41.536, 34.295, 26.758, 19.807, 14.001, 9.599, 6.504, 4.439,
    3.126, 2.370, 2.000])


def t_z(eta):
    """Lee's horizontal-mean profile T_Z (K), clamped at the top of Table 1."""
    x = np.log10(np.maximum(eta, LEE_ETA[-1]))
    return sum(c * x**i for i, c in enumerate(T_Z_COEFFS))


def t_h(eta):
    """Lee's T_H (K), linear in log10(eta), clamped outside Table 1."""
    x = np.log10(eta)
    xp = np.log10(LEE_ETA[::-1])
    return np.interp(x, xp, LEE_T_H[::-1])


def t_z_profile(heights, p_surf, gravity, rd):
    """Temperature (K) and pressure (Pa) at heights (m) for T = T_Z."""
    def rhs(z, y):
        lnp = y[0]
        return [-gravity / (rd * t_z(np.exp(lnp) / p_surf))]

    sol = solve_ivp(rhs, (heights[0], heights[-1]), [np.log(p_surf)],
                    t_eval=heights, rtol=1e-12, atol=1e-12)
    pressure = np.exp(sol.y[0])
    return t_z(pressure / p_surf), pressure


def max_lapse(lat_deg, p_surf, gravity, rd, depth):
    """Largest -dT/dz (K/km) of T_eq at a latitude, and its height (km)."""
    coslat = np.cos(np.radians(lat_deg))

    def temp(p):
        eta = p / p_surf
        return t_z(eta) + t_h(eta) * coslat

    sol = solve_ivp(lambda z, y: [-gravity / (rd * temp(np.exp(y[0])))],
                    (0.0, depth), [np.log(p_surf)],
                    t_eval=np.linspace(0.0, depth, 2001),
                    rtol=1e-10, atol=1e-10)
    t = temp(np.exp(sol.y[0]))
    lapse = -np.gradient(t, sol.t) * 1.0e3
    i = int(np.argmax(lapse))
    return lapse[i], sol.t[i] / 1.0e3


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
    ap.add_argument("--depth", type=float, default=9.0e4,
                    help="height of the top of the table, m (default: 9e4)")
    ap.add_argument("--points", type=int, default=100,
                    help="number of points in the table, at most 100 "
                         "(default: 100)")
    ap.add_argument("--p-surf", type=float, default=9.2e6,
                    help="surface pressure, Pa (default: 9.2e6)")
    ap.add_argument("--gravity", type=float, default=8.87,
                    help="gravitational acceleration, m/s^2 (default: 8.87)")
    ap.add_argument("--rd", type=float, default=191.4,
                    help="specific gas constant, J/kg/K (default: 191.4)")
    ap.add_argument("--cp", type=float, default=900.0,
                    help="namelist cp on which theta is built, J/kg/K "
                         "(default: 900)")
    ap.add_argument("--p-zero", type=float, default=9.2e6,
                    help="reference pressure, Pa (default: 9.2e6)")
    args = ap.parse_args()

    # profile_interp_kernel_mod holds the table in arrays of 100 elements and
    # does not check profile_size, so a longer table overruns them.
    n = args.points
    if not 2 <= n <= 100:
        sys.exit(f"{n} points; the table must have 2 to 100 points")
    heights = np.linspace(0.0, args.depth, n)

    temperature, pressure = t_z_profile(heights, args.p_surf, args.gravity,
                                        args.rd)
    theta = temperature * (args.p_zero / pressure) ** (args.rd / args.cp)

    adiabatic = args.gravity / args.cp * 1.0e3
    report = [f"T {temperature[0]:.2f} -> {temperature[-1]:.2f} K, "
              f"p {pressure[0]:.4E} -> {pressure[-1]:.4E} Pa "
              f"(eta at top {pressure[-1] / args.p_surf:.3E}, "
              f"Table 1 ends at {LEE_ETA[-1]:.3E}), "
              f"theta {theta[0]:.2f} -> {theta[-1]:.2f} K"]
    for lat in (0.0, 90.0):
        lapse, z = max_lapse(lat, args.p_surf, args.gravity, args.rd,
                             args.depth)
        state = "stable" if lapse < adiabatic else "SUPERADIABATIC"
        report.append(f"T_eq at {lat:.0f} deg: largest lapse rate "
                      f"{lapse:.2f} K/km at {z:.1f} km, g/cp {adiabatic:.2f} "
                      f"K/km: {state}")
    print("\n".join(report), file=sys.stderr)

    print("[namelist:initial_temperature]")
    print(rose_list("profile_data", theta))
    print(rose_list("profile_heights", heights))
    print(f"profile_size={n}")


if __name__ == "__main__":
    main()
