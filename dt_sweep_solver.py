#!/usr/bin/env python3
"""Mixed-solver behaviour and cost in a timestep sweep.

Each run directory holds configuration.nml and PET0.gungho_model.Log and is
named dtDT_VARIANT or mode_dtDT_VARIANT (e.g. dt600_on, mode_dt600_on2). The
logs run to millions of steps, so they are streamed and only running totals
are kept. Per run it prints:

  last/end  last timestep started, and timestep_end
  solves    mixed solves logged
  it_mean   mean iterations per mixed solve, (cycle - 1) * gcrk + vectors in
  it_max    the last cycle, from the BLOCK_GCR solver_algorithm lines
  notconv   mixed solves logged as not converged
  limit     transport steps whose substepping was capped
  errors    lines containing ERROR
  s/step    wallclock per timestep run (from timestep_start), from the started
            and finished lines of run.log, when it has them
  h/day     wallclock hours per simulated Earth day (s/step scaled by dt)

Then, per timestep, the ratio of switch-on to switch-off mean iterations, for
runs named on and off.
"""
import argparse
import datetime
import os
import re

SOLVE = re.compile(r"BLOCK_GCR solver_algorithm: \[\s*(\d+),\s*(\d+)\], iters")
STEP = re.compile(r"Start of timestep\s+(\d+)")


def read_log(path, gcrk):
    n, total, it_max = 0, 0, 0
    not_conv = limits = errors = last_step = 0
    prev = None

    def close(cycle, vectors):
        nonlocal n, total, it_max
        iters = (cycle - 1) * gcrk + vectors
        n += 1
        total += iters
        it_max = max(it_max, iters)

    with open(path, errors="replace") as log:
        for line in log:
            m = SOLVE.search(line)
            if m:
                cycle, vectors = int(m.group(1)), int(m.group(2))
                # A solve's cycles count up from 1; a count that does not
                # rise starts the next solve.
                if prev is not None and cycle <= prev[0]:
                    close(*prev)
                prev = (cycle, vectors)
                continue
            m = STEP.search(line)
            if m:
                last_step = int(m.group(1))
            elif "NOT converged" in line:
                not_conv += 1
            elif "Limiting this to" in line:
                limits += 1
            if "ERROR" in line:
                errors += 1
    if prev is not None:
        close(*prev)
    return n, total / n if n else float("nan"), it_max, not_conv, limits, \
        errors, last_step


def wallclock(path):
    """Seconds between the started and finished lines of run.log."""
    stamps = {}
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as log:
        for line in log:
            word = line.split(" ", 1)[0]
            if word in ("started", "finished"):
                stamps[word] = datetime.datetime.strptime(
                    line.split()[1], "%Y-%m-%dT%H:%M:%SZ")
    if len(stamps) < 2:
        return None
    return (stamps["finished"] - stamps["started"]).total_seconds()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories, dtDT_SWITCH")
    args = ap.parse_args()

    means = {}
    print(f"{'run':<16} {'last/end':>17} {'solves':>9} {'it_mean':>7} "
          f"{'it_max':>6} {'notconv':>7} {'limit':>6} {'errors':>6} "
          f"{'s/step':>7} {'h/day':>7}")
    for path in sorted(args.runs):
        name = os.path.basename(os.path.normpath(path))
        m = re.fullmatch(r"(?:mode_)?dt(\d+)_(\w+)", name)
        if not m:
            continue
        conf = open(os.path.join(path, "configuration.nml")).read()
        end = int(re.search(r"^\s*timestep_end='?(\d+)", conf, re.M).group(1))
        start = int(re.search(r"^\s*timestep_start='?(\d+)", conf, re.M).group(1))
        gcrk = int(re.search(r"&mixed_solver.*?^\s*gcrk=(\d+)", conf,
                             re.M | re.S).group(1))
        # Rank 0's log is PET0 or PET00, depending on the rank count
        log = os.path.join(path, "PET0.gungho_model.Log")
        if not os.path.exists(log):
            log = os.path.join(path, "PET00.gungho_model.Log")
        n, mean, it_max, not_conv, limits, errors, last = read_log(log, gcrk)
        dt = int(m.group(1))
        means[(dt, m.group(2))] = mean
        seconds = wallclock(os.path.join(path, "run.log"))
        steps_run = last - start + 1
        s_step = seconds / steps_run if seconds and steps_run > 0 else float("nan")
        h_day = s_step * 86400.0 / dt / 3600.0
        print(f"{name:<16} {last:>8}/{end:<8} {n:9d} {mean:7.2f} {it_max:6d} "
              f"{not_conv:7d} {limits:6d} {errors:6d} {s_step:7.3f} "
              f"{h_day:7.3f}", flush=True)

    print("\nSwitch-on over switch-off mean iterations:")
    for dt in sorted({k[0] for k in means}):
        if (dt, "on") in means and (dt, "off") in means:
            print(f"  dt={dt:5d}  {means[(dt, 'on')] / means[(dt, 'off')]:.3f}")


if __name__ == "__main__":
    main()
