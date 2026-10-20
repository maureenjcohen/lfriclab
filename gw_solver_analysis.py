#!/usr/bin/env python3
"""Decision table for the variable-c_p solver question, from a gravity-wave
slice timestep sweep.

Each run directory holds configuration.nml, run.log and PET0.gungho_model.Log,
and is named BG_dtDT_SWITCH (e.g. venus_dt600_on). The runs log the
conservation diagnostics every step; the background is at rest, so the kinetic
energy is the energy of the waves. Per run it prints:

  done      ran to timestep_end with no ERROR in the log and finite energy
  solves    mixed solves logged
  it_mean   mean iterations per mixed solve, (cycle - 1) * gcrk + vectors in
  it_max    the last cycle, from the BLOCK_GCR solver_algorithm lines
  notconv   mixed solves logged as not converged
  limit     transport steps whose substepping was capped
  growth    slope of ln(kinetic energy) over the second half of the run, per
            1000 s; positive means the waves are gaining energy
  ke_ratio  kinetic energy at the last step over that at the first
  s/step    wallclock per timestep, from run.log

Then, per background and timestep, it compares switch on against switch off,
which uses the solver with a consistent c_p0 and so is the control. A flag
marks switch-on behaviour the control does not show: a failure, more than
20 % more iterations per solve, or energy growing where the control's does
not.
"""
import argparse
import datetime as dt_mod
import os
import re

import numpy as np

SOLVE = re.compile(r"BLOCK_GCR solver_algorithm: \[\s*(\d+),\s*(\d+)\], iters")
STEP = re.compile(r"Start of timestep\s+(\d+)")
KINETIC = re.compile(r"Conservation: (horizontal|vertical) kinetic energy\s+(\S+)")


def namelist_value(text, key):
    m = re.search(rf"^{key}=([^,\n]+)", text, re.M)
    return m.group(1).strip("'") if m else None


def read_log(path, gcrk):
    """Solver iterations, kinetic energy per step, last step and errors."""
    iters, errors, not_conv, limits = [], [], 0, 0
    energy = {}
    last_step, prev = 0, None
    with open(path, errors="replace") as log:
        for line in log:
            m = SOLVE.search(line)
            if m:
                cycle, vectors = int(m.group(1)), int(m.group(2))
                # A solve's cycles count up from 1; a count that does not
                # rise starts the next solve.
                if prev is not None and cycle <= prev[0]:
                    iters.append((prev[0] - 1) * gcrk + prev[1])
                prev = (cycle, vectors)
                continue
            m = KINETIC.search(line)
            if m:
                energy.setdefault(last_step, [0.0, 0.0])
                energy[last_step][m.group(1) == "vertical"] = float(m.group(2))
                continue
            m = STEP.search(line)
            if m:
                last_step = int(m.group(1))
            elif "NOT converged" in line:
                not_conv += 1
            elif "Limiting this to" in line:
                limits += 1
            if "ERROR" in line and len(errors) < 3:
                errors.append(line.strip())
    if prev is not None:
        iters.append((prev[0] - 1) * gcrk + prev[1])
    steps = np.array(sorted(k for k in energy if k > 0))
    kinetic = np.array([sum(energy[k]) for k in steps])
    return np.array(iters), steps, kinetic, last_step, errors, not_conv, limits


def wallclock(path):
    started = finished = rc = None
    with open(path, errors="replace") as log:
        for line in log:
            if line.startswith("started "):
                started = line.split()[1]
            elif line.startswith("finished "):
                finished, rc = line.split()[1], line.split("rc=")[-1].strip()
    if not (started and finished):
        return None, rc
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    span = (dt_mod.datetime.strptime(finished, fmt)
            - dt_mod.datetime.strptime(started, fmt))
    return span.total_seconds(), rc


class Run:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        m = re.fullmatch(r"(\w+?)_dt(\d+)_(on|off)", self.name)
        if not m:
            raise SystemExit(f"{path}: name is not BG_dtDT_SWITCH")
        self.bg, self.dt, self.switch = m.group(1), int(m.group(2)), m.group(3)

        conf = open(os.path.join(path, "configuration.nml")).read()
        self.steps = int(namelist_value(conf, "timestep_end"))
        gcrk = int(re.search(r"&mixed_solver.*?^gcrk=(\d+)", conf,
                             re.M | re.S).group(1))

        (self.iters, self.ke_steps, self.kinetic, self.last_step, self.errors,
         self.not_conv, self.limits) = read_log(
             os.path.join(path, "PET0.gungho_model.Log"), gcrk)
        seconds, self.rc = wallclock(os.path.join(path, "run.log"))
        self.s_per_step = seconds / self.last_step if seconds and self.last_step \
            else np.nan

        self.done = (self.last_step == self.steps and not self.errors
                     and self.kinetic.size > 0
                     and np.isfinite(self.kinetic).all())

    def growth(self):
        """Slope of ln(kinetic energy) over the second half, per 1000 s."""
        times = self.ke_steps * self.dt
        keep = (times >= 0.5 * self.steps * self.dt) & (self.kinetic > 0) \
            & np.isfinite(self.kinetic)
        if keep.sum() < 3:
            return np.nan
        return 1000.0 * np.polyfit(times[keep], np.log(self.kinetic[keep]), 1)[0]

    def ke_ratio(self):
        if self.kinetic.size < 2 or self.kinetic[0] <= 0:
            return np.nan
        return self.kinetic[-1] / self.kinetic[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories, BG_dtDT_SWITCH")
    ap.add_argument("--iter-threshold", type=float, default=0.2,
                    help="fractional excess of switch-on iterations that is "
                         "flagged (default: 0.2)")
    args = ap.parse_args()

    runs = {}
    for path in args.runs:
        run = Run(path)
        runs[(run.bg, run.switch, run.dt)] = run

    print(f"{'run':<18} {'done':>4} {'solves':>6} {'it_mean':>7} {'it_max':>6} "
          f"{'notconv':>7} {'limit':>5} {'growth':>8} {'ke_ratio':>9} "
          f"{'s/step':>7}")
    for key in sorted(runs):
        run = runs[key]
        it_mean = run.iters.mean() if run.iters.size else np.nan
        it_max = run.iters.max() if run.iters.size else 0
        print(f"{run.name:<18} {'yes' if run.done else 'NO':>4} "
              f"{run.iters.size:6d} {it_mean:7.2f} {it_max:6d} "
              f"{run.not_conv:7d} {run.limits:5d} {run.growth():8.3f} "
              f"{run.ke_ratio():9.2e} {run.s_per_step:7.3f}")
        if not run.done:
            print(f"    stopped at step {run.last_step} of {run.steps}, "
                  f"rc={run.rc}")
            for line in run.errors:
                print(f"    {line}")

    print("\nSwitch on against switch off (the control), per timestep:")
    for bg in sorted({k[0] for k in runs}):
        for dt in sorted({k[2] for k in runs if k[0] == bg}):
            on, off = runs.get((bg, "on", dt)), runs.get((bg, "off", dt))
            if not (on and off):
                continue
            flags = []
            if off.done and not on.done:
                flags.append("on fails, off runs")
            if on.iters.size and off.iters.size:
                ratio = on.iters.mean() / off.iters.mean()
                if ratio > 1.0 + args.iter_threshold:
                    flags.append(f"iterations x{ratio:.2f}")
            else:
                ratio = np.nan
            g_on, g_off = on.growth(), off.growth()
            if g_on > 0.0 and not g_off > 0.0:
                flags.append("energy grows, off does not")
            print(f"  {bg:<5} dt={dt:5d}  iter ratio {ratio:5.2f}  "
                  f"growth on {g_on:8.3f} off {g_off:8.3f}  "
                  f"{'; '.join(flags) if flags else 'no flag'}")


if __name__ == "__main__":
    main()
