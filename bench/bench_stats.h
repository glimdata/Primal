/*
 * PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.  A copy of the License
 * is in the repository root (LICENSE) and at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

/* bench_stats.h - shared by the bench drivers: the `--repeat N` option, wall
 * and CPU timing of PRIMAL_optimize, and the key=value tail every driver
 * prints after its positional CSV fields:
 *   wall_med, cpu_min, cpu_med   min and median over the N runs (seconds)
 *   ticks                        PRIMAL_DINF_OPTIMIZER_TICKS, deterministic
 *   intpnt_iter, simplex_iter, mio_nodes, mio_relax   the counters
 * The ticks and the counters are those of the last run; a driver builds a
 * fresh task for every run, so each run is the same cold solve and the
 * deterministic values are identical across runs. */
#ifndef BENCH_STATS_H
#define BENCH_STATS_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "../primal.h"

#define BENCH_MAX_REPEAT 64

typedef struct {
    int n;                                          /* runs recorded */
    double wall[BENCH_MAX_REPEAT], cpu[BENCH_MAX_REPEAT];
    double ticks;
    int intpnt, simplex, nodes, relax;
} BenchStats;

/* Removes `--repeat N` from argv and returns N (1 when absent; N is clamped
 * to 1..BENCH_MAX_REPEAT), so the driver's positional arguments keep their
 * places. */
static inline int bench_repeat(int *argc, char **argv) {
    int n = 1;
    for (int i = 1; i < *argc; i++) {
        if (strcmp(argv[i], "--repeat") == 0 && i + 1 < *argc) {
            n = atoi(argv[i + 1]);
            for (int k = i; k + 2 <= *argc; k++) argv[k] = argv[k + 2];
            *argc -= 2;
            break;
        }
    }
    if (n < 1) n = 1;
    if (n > BENCH_MAX_REPEAT) n = BENCH_MAX_REPEAT;
    return n;
}

/* Current wall-clock time in seconds. */
static inline double bench_wall(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

/* Runs PRIMAL_optimize once, timed by the wall clock and by clock() (process
 * CPU time, every thread included), and records the run's timings, ticks and
 * counters.  Returns the response code. */
static inline PRIMALrescodee bench_optimize(PRIMALtask_t t, BenchStats *s) {
    double w0 = bench_wall();
    clock_t c0 = clock();
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double cpu = (double)(clock() - c0) / (double)CLOCKS_PER_SEC;
    double wall = bench_wall() - w0;
    if (s->n < BENCH_MAX_REPEAT) { s->wall[s->n] = wall; s->cpu[s->n] = cpu; s->n++; }
    int sp = 0, sd = 0;
    PRIMAL_getdouinf(t, PRIMAL_DINF_OPTIMIZER_TICKS, &s->ticks);
    PRIMAL_getintinf(t, PRIMAL_IINF_INTPNT_ITER, &s->intpnt);
    PRIMAL_getintinf(t, PRIMAL_IINF_SIM_PRIMAL_ITER, &sp);
    PRIMAL_getintinf(t, PRIMAL_IINF_SIM_DUAL_ITER, &sd);
    s->simplex = sp + sd;
    PRIMAL_getintinf(t, PRIMAL_IINF_MIO_NUM_SOLVED_NODES, &s->nodes);
    PRIMAL_getintinf(t, PRIMAL_IINF_MIO_NUM_RELAX, &s->relax);
    return rc;
}

static inline int bench_cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Smallest of n timings. */
static inline double bench_min(const double *v, int n) {
    double m = v[0];
    for (int i = 1; i < n; i++) if (v[i] < m) m = v[i];
    return m;
}

/* Median of n timings (the mean of the two middle ones when n is even). */
static inline double bench_median(const double *v, int n) {
    double s[BENCH_MAX_REPEAT];
    memcpy(s, v, (size_t)n * sizeof(double));
    qsort(s, (size_t)n, sizeof(double), bench_cmp);
    return (n % 2) ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/* The key=value tail of a CSV line (no newline). */
static inline void bench_print_tail(const BenchStats *s) {
    printf(",wall_med=%.6f,cpu_min=%.6f,cpu_med=%.6f,ticks=%.6f,"
           "intpnt_iter=%d,simplex_iter=%d,mio_nodes=%d,mio_relax=%d",
           bench_median(s->wall, s->n), bench_min(s->cpu, s->n), bench_median(s->cpu, s->n),
           s->ticks, s->intpnt, s->simplex, s->nodes, s->relax);
}

/* The header names of the tail, for drivers that print a header line. */
#define BENCH_TAIL_HEADER ",wall_med,cpu_min,cpu_med,ticks,intpnt_iter,simplex_iter,mio_nodes,mio_relax"

#endif /* BENCH_STATS_H */
