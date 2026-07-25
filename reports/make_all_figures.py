"""Build the five manuscript figures from their canonical sources.

Figure_1 <- make_fig1.py   Figure_2 <- make_fig2.py   Figure_3 <- make_fig3.py
Figure_4 <- make_fig345.py (Figure 4 block)           Figure_5 <- make_fig5.py

make_fig345.py also contains superseded Figure 3 and Figure 5 blocks; they write to
reports/figs/superseded/ and must not be used for the manuscript.

    export CCEP_ROOT=/path/to/analysis-root
    python make_all_figures.py [--out DIR] [--only 1 3]
"""
import argparse
import os
import runpy
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

CANONICAL = [
    ("Figure_1.png", "make_fig1.py"),
    ("Figure_2.png", "make_fig2.py"),
    ("Figure_3.png", "make_fig3.py"),
    ("Figure_4.png", "make_fig345.py"),
    ("Figure_5.png", "make_fig5.py"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="also copy the finished PNGs here")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="build only these figure numbers")
    args = ap.parse_args()

    if "CCEP_ROOT" not in os.environ:
        sys.exit("CCEP_ROOT is not set; point it at the analysis root "
                 "(data/processed, data/traces, reports/).")

    sys.path.insert(0, str(HERE))
    os.chdir(HERE)

    built = []
    for name, script in CANONICAL:
        if args.only and int(name.split("_")[1][0]) not in args.only:
            continue
        print(f"\n=== {name} <- {script} " + "=" * 30)
        t0 = time.time()
        try:
            runpy.run_path(str(HERE / script), run_name="__main__")
        except Exception as exc:
            print(f"  !! {script} failed: {type(exc).__name__}: {exc}")
            continue
        print(f"  done in {time.time() - t0:.1f}s")
        built.append(name)

    figs = Path(os.environ["CCEP_ROOT"]) / "reports" / "figs"
    print("\n=== summary " + "=" * 40)
    for name, _ in CANONICAL:
        p = figs / name
        stamp = time.strftime("%H:%M:%S", time.localtime(p.stat().st_mtime)) if p.exists() else "MISSING"
        print(f"  {name:14s} {stamp}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name in built:
            src = figs / name
            if src.exists():
                shutil.copy2(src, args.out / name)
                print(f"  copied {name} -> {args.out}")


if __name__ == "__main__":
    main()
