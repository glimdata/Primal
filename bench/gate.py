#!/usr/bin/env python3
#
# PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
# Copyright 2026 Gaetano Minardi
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  A copy of the License
# is in the repository root (LICENSE) and at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations
# under the License.

"""Dominance gate over one route's benchmark corpus.  Standard library only.

    python3 bench/gate.py ROUTE --baseline [--repeat N]
        runs the corpus and writes research/baselines/ROUTE.csv
    python3 bench/gate.py ROUTE [--repeat N] [--identical-counts] [--obj-tol T]
        runs the corpus, diffs it against the stored baseline, prints one line
        per instance and exits 1 when the gate fails

Routes and their corpora (the drivers are built by `make bench`):
    lp     the lp and netlib rows of bench/instances/index.csv (solve_mps)
    qp     the qp rows                                          (solve_mps)
    milp   the milp rows                                        (solve_mps)
    socp   the SOCP family of conic_bench, n = 40, 120, 200
    sdp    the SDP family of sdp_sweep, d = 4..22

Every instance is solved N times (default 3) on a fresh task; the CSV keeps
the deterministic ticks and counters and the median and minimum CPU time.

The rules, per instance:
    status     a solved baseline instance must still solve (failure to success
               is allowed, never the reverse)
    objective  when both solve, |run - base| <= T * (1 + |base|), T = 1e-6
    ticks      the run's ticks are not higher than the baseline's
    cpu        the run's median CPU time is within the noise band of the
               baseline's: 5%, or the measured run-to-run spread of either
               side (median over minimum) when larger, and never less than
               one millisecond
    counts     iterations, nodes and relaxations are reported; with
               --identical-counts every one of them must equal the baseline
and, per route, the shifted geometric mean (shift 1) of CPU seconds and of
ticks over the instances that solved on both sides.
"""
import argparse
import csv
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH_OUT = os.path.join(ROOT, "out", "bench")
INSTANCES = os.path.join(HERE, "instances")
BASELINES = os.path.join(ROOT, "research", "baselines")

FIELDS = ["model", "status", "objective", "ticks", "intpnt_iter", "simplex_iter",
          "nodes", "relax", "cpu", "cpu_min"]
COUNTS = ["intpnt_iter", "simplex_iter", "nodes", "relax"]
ROUTES = {"lp": ("lp", "netlib"), "qp": ("qp",), "milp": ("milp",), "socp": (), "sdp": ()}
SDP_RANGE = (4, 22)
TICK_SLACK = 1e-9          # floating-point slack on an integer-valued sum
CPU_BAND = 0.05
CPU_FLOOR = 1e-3


def parse_tail(tokens):
    """The key=value tokens of a driver line as a dict of floats."""
    out = {}
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = float(v)
    return out


def record(model, rc, obj, tail):
    return {"model": model, "status": int(rc), "objective": float(obj),
            "ticks": tail["ticks"], "intpnt_iter": int(tail["intpnt_iter"]),
            "simplex_iter": int(tail["simplex_iter"]), "nodes": int(tail["mio_nodes"]),
            "relax": int(tail["mio_relax"]), "cpu": tail["cpu_med"], "cpu_min": tail["cpu_min"]}


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3600).stdout
    except subprocess.TimeoutExpired:
        sys.exit("timed out: %s" % " ".join(cmd))


def run_mps(classes, repeat):
    exe = os.path.join(BENCH_OUT, "solve_mps")
    rows = []
    with open(os.path.join(INSTANCES, "index.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["class"] not in classes:
                continue
            path = os.path.join(INSTANCES, r["name"] + ".mps")
            out = run([exe, "--repeat", str(repeat), path]).strip()
            p = out.split(",")
            if p[0] == "READ_ERROR" or len(p) < 5:
                sys.exit("%s: %s" % (r["name"], out))
            rows.append(record(r["name"], p[0], p[3], parse_tail(p[5:])))
    return rows


def run_socp(repeat):
    exe = os.path.join(BENCH_OUT, "conic_bench")
    rows = []
    for ln in run([exe, "--repeat", str(repeat)]).strip().splitlines()[1:]:
        p = ln.split(",")
        if p[0] != "socp":
            continue
        rows.append(record("socp_%s" % p[1], p[5].split("=")[1], p[3], parse_tail(p[6:])))
    return rows


def run_sdp(repeat):
    exe = os.path.join(BENCH_OUT, "sdp_sweep")
    rows = []
    cmd = [exe, "--repeat", str(repeat), str(SDP_RANGE[0]), str(SDP_RANGE[1])]
    for ln in run(cmd).strip().splitlines()[1:]:
        p = ln.split(",")
        rows.append(record("sdp_%s" % p[0], p[3], p[1], parse_tail(p[5:])))
    return rows


def run_route(route, repeat):
    for exe in ("solve_mps", "conic_bench", "sdp_sweep"):
        if not os.path.exists(os.path.join(BENCH_OUT, exe)):
            sys.exit("run `make bench` first (%s missing)" % exe)
    if route == "socp":
        return run_socp(repeat)
    if route == "sdp":
        return run_sdp(repeat)
    return run_mps(ROUTES[route], repeat)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path):
    with open(path) as fh:
        rows = []
        for r in csv.DictReader(fh):
            rows.append({k: (r[k] if k == "model" else
                             int(r[k]) if k in COUNTS or k == "status" else float(r[k]))
                         for k in FIELDS})
        return rows


def sgm(values, shift=1.0):
    """Shifted geometric mean, the corpus summary that no single instance dominates."""
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


def spread(r):
    return (r["cpu"] - r["cpu_min"]) / r["cpu_min"] if r["cpu_min"] > 0 else 0.0


def compare(base, cur, obj_tol, identical_counts):
    """One verdict line per instance; returns (lines, failed, summary)."""
    lines, failed = [], False
    bmap = {r["model"]: r for r in base}
    cmap = {r["model"]: r for r in cur}
    cpu_b, cpu_c, tk_b, tk_c = [], [], [], []
    for name in [r["model"] for r in base] + [r["model"] for r in cur if r["model"] not in bmap]:
        b, c = bmap.get(name), cmap.get(name)
        if b is None:
            lines.append("%-14s new instance, not in the baseline" % name)
            continue
        if c is None:
            lines.append("%-14s MISSING from the run" % name)
            failed = True
            continue
        problems = []
        if b["status"] == 0 and c["status"] != 0:
            problems.append("status %d -> %d" % (b["status"], c["status"]))
        both_ok = b["status"] == 0 and c["status"] == 0
        if both_ok and abs(c["objective"] - b["objective"]) > obj_tol * (1.0 + abs(b["objective"])):
            problems.append("objective %.10g -> %.10g" % (b["objective"], c["objective"]))
        if c["ticks"] > b["ticks"] * (1.0 + TICK_SLACK):
            problems.append("ticks up %.3f -> %.3f" % (b["ticks"], c["ticks"]))
        band = max(CPU_BAND, spread(b), spread(c))
        allowed = max(b["cpu"] * (1.0 + band), b["cpu"] + CPU_FLOOR)
        if c["cpu"] > allowed:
            problems.append("cpu %.4f -> %.4f (band %.0f%%)" % (b["cpu"], c["cpu"], 100 * band))
        diffs = [k for k in COUNTS if b[k] != c[k]]
        if diffs and identical_counts:
            problems.append("counts differ: " + ", ".join("%s %d -> %d" % (k, b[k], c[k]) for k in diffs))
        if both_ok:
            cpu_b.append(b["cpu"]); cpu_c.append(c["cpu"]); tk_b.append(b["ticks"]); tk_c.append(c["ticks"])
        verdict = "FAIL " + "; ".join(problems) if problems else "ok"
        if diffs and not identical_counts:
            verdict += " (counts " + ", ".join("%s %d -> %d" % (k, b[k], c[k]) for k in diffs) + ")"
        lines.append("%-14s rc %d->%d  ticks %10.3f -> %10.3f (x%.3f)  cpu %.4f -> %.4f (x%.3f)  %s"
                     % (name, b["status"], c["status"], b["ticks"], c["ticks"],
                        c["ticks"] / b["ticks"] if b["ticks"] > 0 else float("nan"),
                        b["cpu"], c["cpu"], c["cpu"] / b["cpu"] if b["cpu"] > 0 else float("nan"),
                        verdict))
        failed = failed or bool(problems)
    summary = ("shifted geometric mean (shift 1) over %d solved instances: "
               "cpu %.4f -> %.4f s (x%.3f), ticks %.3f -> %.3f (x%.3f)"
               % (len(cpu_b), sgm(cpu_b), sgm(cpu_c),
                  sgm(cpu_c) / sgm(cpu_b) if sgm(cpu_b) > 0 else float("nan"),
                  sgm(tk_b), sgm(tk_c), sgm(tk_c) / sgm(tk_b) if sgm(tk_b) > 0 else float("nan")))
    return lines, failed, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("route", choices=sorted(ROUTES))
    ap.add_argument("--baseline", action="store_true", help="write the route's baseline CSV instead of diffing")
    ap.add_argument("--repeat", type=int, default=3, help="solves per instance (default 3)")
    ap.add_argument("--identical-counts", action="store_true", help="require every counter equal to the baseline")
    ap.add_argument("--obj-tol", type=float, default=1e-6, help="relative objective tolerance (default 1e-6)")
    ap.add_argument("--baseline-dir", default=BASELINES, help="where ROUTE.csv lives (default research/baselines)")
    args = ap.parse_args()
    path = os.path.join(args.baseline_dir, args.route + ".csv")
    rows = run_route(args.route, args.repeat)
    if args.baseline:
        write_csv(path, rows)
        for r in rows:
            print("%-14s rc %d  obj %.10g  ticks %10.3f  cpu %.4f  counts %d/%d/%d/%d"
                  % (r["model"], r["status"], r["objective"], r["ticks"], r["cpu"],
                     r["intpnt_iter"], r["simplex_iter"], r["nodes"], r["relax"]))
        print("baseline written: %s (%d instances)" % (path, len(rows)))
        return 0
    if not os.path.exists(path):
        sys.exit("no baseline for route %s: run with --baseline first (%s)" % (args.route, path))
    lines, failed, summary = compare(read_csv(path), rows, args.obj_tol, args.identical_counts)
    for ln in lines:
        print(ln)
    print(summary)
    print("gate %s" % ("FAILED" if failed else "passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
