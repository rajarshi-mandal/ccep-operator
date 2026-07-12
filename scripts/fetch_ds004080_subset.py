#!/usr/bin/env python
"""Selective fetch for ds004080 (ccepAge) — download only the LARGEST single run per subject.

ds004080 has ~10 SPES runs/subject with multi-GB BrainVision .eeg each (~20GB/subject, ~1.5TB total
— infeasible to pull whole). Each run stimulates a subset of sites, so the largest run (most
stim events) gives the best single-run site count. We download just that run's signal + sidecars +
the subject's electrodes/coordsystem into a clean per-subject layout, so the existing single-run
build_subject() works unchanged.

Robust + resumable: skips files already present at the correct size; retries each download a few
times; never aborts the whole run on one failed/again-flaky file.

Usage: ../.venv/bin/python scripts/fetch_ds004080_subset.py N   (first N subjects; default 20)
"""
import os
import subprocess
import sys

S3 = "s3://openneuro.org/ds004080"
DEST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Open Neuro ds004080"))


def s3_ls(path):
    out = subprocess.run(["aws", "s3", "ls", "--no-sign-request", path],
                         capture_output=True, text=True).stdout
    return out.splitlines()


def s3_ls_recursive(path):
    out = subprocess.run(["aws", "s3", "ls", "--no-sign-request", "--recursive", path],
                         capture_output=True, text=True).stdout
    return [ln.split() for ln in out.splitlines() if ln.strip()]


def cp(src, dst, expected_size, tries=4):
    """Download src->dst unless already present at expected_size; retry; return True on success."""
    if os.path.exists(dst) and os.path.getsize(dst) == int(expected_size):
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for _ in range(tries):
        r = subprocess.run(["aws", "s3", "cp", "--no-sign-request",
                            "--cli-read-timeout", "0", src, dst],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) == int(expected_size):
            return True
    return False


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    subs = [ln.split()[-1].rstrip("/") for ln in s3_ls(f"{S3}/") if "PRE sub-" in ln][:n]
    print(f"fetching largest run for {len(subs)} subjects")
    for sub in subs:
        files = s3_ls_recursive(f"{S3}/{sub}/")          # rows: [date, time, size, key]
        size_of = {f[3]: int(f[2]) for f in files}
        # subject-level electrodes/coordsystem (shared)
        for key, sz in size_of.items():
            base = key.split("/")[-1]
            if base.endswith(("_electrodes.tsv", "_electrodes.json", "_coordsystem.json")):
                cp(f"s3://openneuro.org/{key}", os.path.join(DEST, key.split("ds004080/")[1]), sz)
        # pick run with the largest events.tsv (most stims)
        ev = [(sz, key) for key, sz in size_of.items() if key.endswith("_events.tsv")]
        if not ev:
            print(f"  {sub}: no events"); continue
        _, ev_key = max(ev)
        stem = ev_key[: ev_key.index("_events.tsv")].split("/")[-1]   # sub-..._run-XXXXXX
        ok, total = True, 0
        for key, sz in size_of.items():
            if stem in key.split("/")[-1]:
                got = cp(f"s3://openneuro.org/{key}", os.path.join(DEST, key.split("ds004080/")[1]), sz)
                ok &= got
                total += sz
        print(f"  {sub}: run {stem.split('_')[-1]} ({total/1e9:.1f} GB){'' if ok else '  [INCOMPLETE]'}")


if __name__ == "__main__":
    main()
