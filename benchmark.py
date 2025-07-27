#!/usr/bin/env python3
import argparse
import ctypes
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()

def build_shared_lib(force=False, verbose=True):
    """
    Build the shared library for the C batch L2 implementation.
    Assumes GCC/Clang on Linux/macOS. Windows users can compile to a DLL manually (see l2.c).
    """
    libname = {
        "Linux": "libl2.so",
        "Darwin": "libl2.dylib",
        "Windows": "l2.dll",
    }.get(platform.system(), "libl2.so")

    libpath = HERE / libname
    if libpath.exists() and not force:
        if verbose:
            print(f"[build] Using existing {libpath}")
        return libpath

    if platform.system() == "Windows":
        raise RuntimeError(
            "Auto-build is only set up for Linux/macOS. "
            "On Windows, compile l2.c to l2.dll yourself (e.g., with MSVC or Mingw-w64) "
            "and place it next to benchmark.py."
        )

    cc = os.environ.get("CC", "cc")
    cflags = ["-O3", "-march=native", "-ffast-math", "-fPIC"]
    if platform.system() == "Darwin":
        ldflags = ["-shared", "-undefined", "dynamic_lookup"]
    else:
        ldflags = ["-shared"]

    cmd = [cc, "l2.c", "-o", str(libpath)] + cflags + ldflags
    if verbose:
        print("[build] ", " ".join(cmd))
    subprocess.check_call(cmd, cwd=HERE)
    return libpath


def load_c_lib(libpath):
    lib = ctypes.CDLL(str(libpath))
    # void l2_batch(const float* a, const float* b, int n_dim, int n_vecs, float* out);
    lib.l2_batch.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.l2_batch.restype = None
    return lib


def l2_python(vec_a, vec_b):
    """Pure-Python L2 distance between two Python lists of floats."""
    s = 0.0
    for x, y in zip(vec_a, vec_b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def time_block(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def summarize_times(times, label, n_ops=None):
    mean_ = statistics.mean(times)
    stdev_ = statistics.stdev(times) if len(times) > 1 else 0.0
    min_ = min(times)
    max_ = max(times)
    throughput = None
    if n_ops is not None:
        throughput = n_ops / mean_
    return {
        "label": label,
        "mean": mean_,
        "stdev": stdev_,
        "min": min_,
        "max": max_,
        "throughput": throughput,
    }


def pretty_print(results, baseline_label):
    # Find baseline for speedup
    baseline = next(r for r in results if r["label"] == baseline_label)
    base_mean = baseline["mean"]

    print("\n=== Benchmark Results ===")
    print(f"Vectors per trial : {args.n_vecs}")
    print(f"Dimensionality    : {args.dim}")
    print(f"Trials (timed)    : {args.trials}")
    print()

    header = [
        "Impl",
        "mean (s)",
        "stdev (s)",
        "min (s)",
        "max (s)",
        "speedup vs " + baseline_label,
        "throughput (vec/s)",
    ]
    widths = [max(len(h), 22) for h in header]
    row_fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(row_fmt.format(*header))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    for r in results:
        speedup = baseline["mean"] / r["mean"]
        thr = r["throughput"]
        print(
            row_fmt.format(
                r["label"],
                f"{r['mean']:.6f}",
                f"{r['stdev']:.6f}",
                f"{r['min']:.6f}",
                f"{r['max']:.6f}",
                f"{speedup:.2f}x",
                f"{thr:,.0f}" if thr is not None else "—",
            )
        )
    print()


def run(args):
    # ---------------------------
    # Generate data outside timing
    # ---------------------------
    rng = np.random.default_rng(args.seed)
    A = rng.standard_normal(size=(args.n_vecs, args.dim), dtype=np.float32)
    B = rng.standard_normal(size=(args.n_vecs, args.dim), dtype=np.float32)

    # Copies for Python-pure lists (so Python timings don't include conversion)
    A_py = [row.tolist() for row in A]
    B_py = [row.tolist() for row in B]

    results = []

    # ---------------------------
    # C implementation (batch)
    # ---------------------------
    if args.run_c:
        libpath = build_shared_lib(force=args.rebuild, verbose=not args.quiet)
        clib = load_c_lib(libpath)

        # Allocate output buffer
        out = np.empty(args.n_vecs, dtype=np.float32)

        # Warmups
        for _ in range(args.warmup):
            clib.l2_batch(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(args.dim),
                ctypes.c_int(args.n_vecs),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )

        def bench_c():
            clib.l2_batch(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(args.dim),
                ctypes.c_int(args.n_vecs),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )

        c_times = time_block(bench_c, args.trials)
        results.append(
            summarize_times(
                c_times, label="C batch (ctypes)", n_ops=args.n_vecs
            )
        )

    # ---------------------------
    # NumPy implementation
    # ---------------------------
    if args.run_numpy:
        # Warmups
        for _ in range(args.warmup):
            _ = np.linalg.norm(A - B, axis=1)

        def bench_numpy():
            _ = np.linalg.norm(A - B, axis=1)

        np_times = time_block(bench_numpy, args.trials)
        results.append(
            summarize_times(
                np_times, label="NumPy (vectorized)", n_ops=args.n_vecs
            )
        )

    # ---------------------------
    # Pure Python implementation
    # ---------------------------
    if args.run_python:
        # Warmups
        for _ in range(args.warmup):
            for i in range(args.n_vecs):
                _ = l2_python(A_py[i], B_py[i])

        def bench_py():
            for i in range(args.n_vecs):
                _ = l2_python(A_py[i], B_py[i])

        py_times = time_block(bench_py, args.trials)
        results.append(
            summarize_times(
                py_times, label="Pure Python", n_ops=args.n_vecs
            )
        )

    # ---------------------------
    # Print report
    # ---------------------------
    if not results:
        print("No benchmarks were selected. See --help.")
        return

    # Choose baseline heuristically: prefer Pure Python, else NumPy, else first.
    baseline_label = "Pure Python"
    labels = [r["label"] for r in results]
    if baseline_label not in labels:
        baseline_label = "NumPy (vectorized)" if "NumPy (vectorized)" in labels else results[0]["label"]

    pretty_print(results, baseline_label=baseline_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2 distance C vs Python benchmark (excluding RNG time).")
    parser.add_argument("--n-vecs", type=int, default=100000,
                        help="Number of vector pairs to benchmark.")
    parser.add_argument("--dim", type=int, default=512,
                        help="Vector dimensionality.")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of timed trials.")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup runs (not timed).")
    parser.add_argument("--seed", type=int, default=12345,
                        help="RNG seed (for reproducibility).")
    parser.add_argument("--no-c", dest="run_c", action="store_false", default=True,
                        help="Disable running the C benchmark.")
    parser.add_argument("--no-numpy", dest="run_numpy", action="store_false", default=True,
                        help="Disable running the NumPy benchmark.")
    parser.add_argument("--no-python", dest="run_python", action="store_false", default=True,
                        help="Disable running the pure Python benchmark.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuilding the C shared library.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress build chatter.")

    args = parser.parse_args()
    run(args)
