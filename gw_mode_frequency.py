#!/usr/bin/env python3
"""Frequency and damping of the standing gravity-wave mode, per run, against
the analytic frequency.

Each run directory holds configuration.nml and PET0.gungho_model.Log, and is
named mode_dtDT_VARIANT (e.g. mode_dt600_on2). The wave starts from rest, so
its kinetic energy goes as

  KE(t) = a exp(-2 gamma t) sin^2(omega t + phi) + c

and a fit of that to the per-step kinetic energy in the log gives the wave
frequency omega and the amplitude damping rate gamma. The fit quality columns
are

  resid_%   rms of the fit residual, as a % of the fitted oscillation's mean
            amplitude a/2
  sd_om_%   standard error of omega from the fit, as a % of omega
  sd_damp   standard error of the damping rate, 1/day
  res_peak  angular frequency of the largest peak in the residual's spectrum,
            in units of the analytic omega; a second wave mode in the run
            shows here, since a single mode leaves only noise

With --plot DIR, each run's kinetic energy, fit, residual and residual
spectrum go in DIR/RUN.png.

The analytic frequency is the gravity-wave root of the dispersion relation for
an isothermal compressible atmosphere between rigid lids,

  omega^4 - omega^2 (c^2 (k^2 + m^2) + c^2/(4 H^2)) + c^2 N^2 k^2 = 0

with k = 2 pi/wavelength, m = pi/depth, H = R T/g the scale height,
N^2 = g^2/(c_p T) and c^2 = gamma_gas R T the sound speed squared, where
gamma_gas = c_p/(c_p - R). c_p is the true heat capacity c_p(T).

Per timestep it prints each variant's frequency error against the analytic
value, and the two differences that decide the solver question:

  on2 - onN  what the linear solver loses by leaving the c_p term out; a
             solver term could at best remove it (N, the most outer
             iterations among the runs)
  ref - an   the error of the time scheme with a consistent solver
"""
import argparse
import os
import re
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from gw_solver_analysis import read_log


def analytic_frequency(t, cp, rd, gravity, wavelength, depth):
    """Gravity-wave frequency (1/s) of the lowest vertical mode."""
    k = 2.0 * np.pi / wavelength
    m = np.pi / depth
    scale_height = rd * t / gravity
    n2 = gravity**2 / (cp * t)
    c2 = cp / (cp - rd) * rd * t
    b = c2 * (k**2 + m**2) + c2 / (4.0 * scale_height**2)
    d = c2 * n2 * k**2
    return np.sqrt(0.5 * (b - np.sqrt(b**2 - 4.0 * d)))


def ke_model(t, a, gamma, omega, phi, c):
    return a * np.exp(-2.0 * gamma * t) * np.sin(omega * t + phi)**2 + c


def spectral_frequency(times, kinetic, omega_guide):
    """Wave frequency (1/s) from the peak of the kinetic energy spectrum, which
    lies at twice the wave frequency. The peak is sought within 50 % of twice
    omega_guide, away from the low frequencies of the decaying envelope."""
    spacing = times[1] - times[0]
    # Zero padding refines the peak well below the spectral resolution of the
    # run, so the fit starts inside the right minimum.
    n = 16 * kinetic.size
    amplitude = np.abs(np.fft.rfft(kinetic - kinetic.mean(), n))
    freqs = 2.0 * np.pi * np.fft.rfftfreq(n, spacing)
    window = np.abs(freqs / (2.0 * omega_guide) - 1.0) < 0.5
    return 0.5 * freqs[window][np.argmax(amplitude[window])]


def fit_mode(times, kinetic, omega_guide):
    """Fitted parameters and their standard errors (nan or inf where the fit
    cannot estimate them)."""
    omega0 = spectral_frequency(times, kinetic, omega_guide)
    scale = kinetic.max()
    # Fit in time scaled by the starting frequency and energy scaled by its
    # maximum, so that all parameters are of order one.
    p0 = [2.0, 0.0, 1.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(ke_model, times * omega0, kinetic / scale,
                               p0=p0, maxfev=20000)
    with np.errstate(invalid="ignore"):
        perr = np.sqrt(np.diag(pcov))
    units = np.array([scale, omega0, omega0, 1.0, scale])
    return popt * units, perr * units


def residual_spectrum(times, residual):
    """Angular frequencies (1/s) and amplitude spectrum of the residual,
    which is sampled once per timestep."""
    spacing = times[1] - times[0]
    amplitude = np.abs(np.fft.rfft(residual - residual.mean())) \
        * 2.0 / residual.size
    freqs = 2.0 * np.pi * np.fft.rfftfreq(residual.size, spacing)
    return freqs[1:], amplitude[1:]


def plot_run(path, name, times, kinetic, fitted, freqs, amplitude, omega_an):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = times / 86400.0
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    axes[0].plot(days, kinetic, lw=0.8, label="log")
    axes[0].plot(days, fitted, lw=0.8, ls="--", label="fit")
    axes[0].set_xlabel("time (days)")
    axes[0].set_ylabel("kinetic energy")
    axes[0].legend()
    axes[1].plot(days, kinetic - fitted, lw=0.8)
    axes[1].set_xlabel("time (days)")
    axes[1].set_ylabel("residual")
    axes[2].plot(freqs / omega_an, amplitude, lw=0.8)
    axes[2].set_xlim(0.0, min(10.0, freqs[-1] / omega_an))
    axes[2].set_xlabel("angular frequency / analytic omega")
    axes[2].set_ylabel("residual amplitude")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(os.path.join(path, name + ".png"), dpi=120)
    plt.close(fig)


class Run:
    def __init__(self, path):
        self.name = os.path.basename(os.path.normpath(path))
        m = re.fullmatch(r"mode_dt(\d+)_(ref|on\d+)", self.name)
        if not m:
            raise SystemExit(f"{path}: name is not mode_dtDT_VARIANT")
        self.dt, self.variant = int(m.group(1)), m.group(2)
        conf = open(os.path.join(path, "configuration.nml")).read()
        self.steps = int(re.search(r"^timestep_end='?(\d+)", conf,
                                   re.M).group(1))
        gcrk = int(re.search(r"&mixed_solver.*?^gcrk=(\d+)", conf,
                             re.M | re.S).group(1))
        (_, ke_steps, self.kinetic, self.last_step, self.errors, _,
         _) = read_log(os.path.join(path, "PET0.gungho_model.Log"), gcrk)
        self.times = ke_steps * float(self.dt)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories, mode_dtDT_VARIANT")
    ap.add_argument("--temperature", type=float, default=700.0,
                    help="background temperature, K (default: 700)")
    ap.add_argument("--cp", type=float, default=1158.2947,
                    help="true heat capacity c_p(T), J/kg/K (default: 1158.2947)")
    ap.add_argument("--rd", type=float, default=191.4,
                    help="specific gas constant, J/kg/K (default: 191.4)")
    ap.add_argument("--gravity", type=float, default=8.87,
                    help="gravitational acceleration, m/s^2 (default: 8.87)")
    ap.add_argument("--wavelength", type=float, default=3.2e6,
                    help="horizontal wavelength, m (default: 3.2e6)")
    ap.add_argument("--depth", type=float, default=9.0e4,
                    help="model depth, m (default: 9e4)")
    ap.add_argument("--plot", metavar="DIR",
                    help="write a figure per run to DIR")
    args = ap.parse_args()

    omega_an = analytic_frequency(args.temperature, args.cp, args.rd,
                                  args.gravity, args.wavelength, args.depth)
    print(f"analytic omega {omega_an:.6e} 1/s, period "
          f"{2 * np.pi / omega_an / 3600:.4f} h\n")

    if args.plot:
        os.makedirs(args.plot, exist_ok=True)
    print(f"{'run':<18} {'done':>4} {'omega':>12} {'period_h':>9} "
          f"{'err_%':>7} {'omega*dt':>8} {'damp_1/d':>9} {'resid_%':>7} "
          f"{'sd_om_%':>7} {'sd_damp':>7} {'res_peak':>8}")
    results = {}
    for path in args.runs:
        run = Run(path)
        done = run.last_step == run.steps and not run.errors
        if run.kinetic.size < 20:
            print(f"{run.name:<18} {'NO':>4}  too few steps logged")
            continue
        popt, perr = fit_mode(run.times, run.kinetic, omega_an)
        omega, gamma = abs(popt[2]), popt[1]
        results[(run.dt, run.variant)] = omega
        fitted = ke_model(run.times, *popt)
        residual = run.kinetic - fitted
        resid = 100 * np.sqrt(np.mean(residual**2)) / (0.5 * abs(popt[0]))
        freqs, amplitude = residual_spectrum(run.times, residual)
        peak = freqs[np.argmax(amplitude)] / omega_an
        print(f"{run.name:<18} {'yes' if done else 'NO':>4} {omega:12.6e} "
              f"{2 * np.pi / omega / 3600:9.4f} "
              f"{100 * (omega / omega_an - 1):7.3f} {omega * run.dt:8.3f} "
              f"{gamma * 86400:9.4f} {resid:7.2f} "
              f"{100 * perr[2] / omega:7.3f} {perr[1] * 86400:7.4f} "
              f"{peak:8.3f}")
        if args.plot:
            plot_run(args.plot, run.name, run.times, run.kinetic, fitted,
                     freqs, amplitude, omega_an)

    print("\nFrequency differences, % of the analytic frequency:")
    for dt in sorted({k[0] for k in results}):
        variants = {k[1]: v for k, v in results.items() if k[0] == dt}
        ons = sorted((int(v[2:]), v) for v in variants if v.startswith("on"))
        line = f"  dt={dt:5d}"
        if "on2" in variants and ons and ons[-1][0] > 2:
            top = ons[-1][1]
            line += (f"  on2 - {top} "
                     f"{100 * (variants['on2'] - variants[top]) / omega_an:7.3f}")
        if "ref" in variants:
            line += f"  ref - an {100 * (variants['ref'] / omega_an - 1):7.3f}"
        print(line)


if __name__ == "__main__":
    main()
