#!/usr/bin/env python3

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

BALANCE_RATIO = 2.0
IDENTITY_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
FOLD_COUNTS = [5, 4]
MIN_ROWS_FOR_SPLIT = 8

def assign_folds(
    csv_path: Path,
    splits_dir: Path,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    csv_path = Path(csv_path).resolve()
    splits_dir = Path(splits_dir)
    stem = csv_path.stem

    def _fallback(reason: str) -> dict[str, Any]:
        if verbose:
            print(f"[seqsplit] {stem}: {reason} → random CV fallback", file=sys.stderr)
        return {
            "success": False,
            "csv_path": csv_path,
            "split_col": None,
            "n_folds": None,
            "seq_id": None,
        }

    seqs: list[tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        lower_to_field = {f.strip().lower(): f for f in (reader.fieldnames or [])}
        name_col = lower_to_field["name"]
        heavy_col = lower_to_field["heavy"]
        light_col = lower_to_field["light"]
        for row in reader:
            name = row[name_col].strip()
            heavy = row[heavy_col].strip()
            light = row[light_col].strip()
            seqs.append((name, heavy + light))

    n = len(seqs)
    if n < MIN_ROWS_FOR_SPLIT:
        return _fallback(f"too few rows ({n} < {MIN_ROWS_FOR_SPLIT})")

    splits_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = splits_dir / f"{stem}.fasta"
    with fasta_path.open("w", encoding="utf-8", newline="\n") as f:
        for name, seq in seqs:
            f.write(f">{name}\n{seq}\n")

    try:
        for seq_id in IDENTITY_THRESHOLDS:
            id_tag = f"{int(seq_id * 100):03d}"
            mmseqs_prefix = splits_dir / f"{stem}_mmseqs_{id_tag}"
            tmp_dir = splits_dir / f"_tmp_{stem}_{id_tag}"
            cluster_tsv: Path | None = None
            try:
                tmp_dir.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(
                    [
                        "mmseqs", "easy-cluster",
                        str(fasta_path),
                        str(mmseqs_prefix),
                        str(tmp_dir),
                        "--min-seq-id", str(seq_id),
                        "-c", "0.8",
                        "--cov-mode", "0",
                        "--cluster-mode", "1",
                        "-v", "0",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"mmseqs easy-cluster failed (exit {result.returncode}):\n"
                        f"{result.stderr[:500]}"
                    )
                cluster_tsv = Path(f"{mmseqs_prefix}_cluster.tsv")
                if not cluster_tsv.is_file():
                    raise RuntimeError(f"Expected mmseqs output not found: {cluster_tsv}")
            except RuntimeError as e:
                if verbose:
                    print(
                        f"[seqsplit] {stem}: id={seq_id:.2f} mmseqs error: {e}",
                        file=sys.stderr,
                    )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if cluster_tsv is None:
                for path in mmseqs_prefix.parent.glob(f"{mmseqs_prefix.name}*"):
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
                continue

            clusters: dict[str, set[str]] = defaultdict(set)
            with cluster_tsv.open(newline="", encoding="utf-8") as fp:
                for row in csv.reader(fp, delimiter="\t"):
                    if not row or len(row) < 2:
                        continue
                    rep, member = row[0].strip(), row[1].strip()
                    if rep and member:
                        clusters[rep].add(member)
            clusters = dict(clusters)
            for path in mmseqs_prefix.parent.glob(f"{mmseqs_prefix.name}*"):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
            n_clusters = len(clusters)
            if verbose:
                print(
                    f"[seqsplit] {stem}: id={seq_id:.2f} → {n_clusters} cluster(s)",
                    file=sys.stderr,
                )

            for k in FOLD_COUNTS:
                if n_clusters < k:
                    continue
                collapsed = deepcopy(clusters)
                while len(collapsed) > k:
                    if len(collapsed) < 2:
                        break
                    reps = sorted(collapsed, key=lambda r: (len(collapsed[r]), r))
                    r1, r2 = reps[0], reps[1]
                    m1, m2 = collapsed.pop(r1), collapsed.pop(r2)
                    merged = m1 | m2
                    if len(m1) > len(m2):
                        new_rep = r1
                    elif len(m2) > len(m1):
                        new_rep = r2
                    else:
                        new_rep = r1 if r1 < r2 else r2
                    collapsed[new_rep] = merged

                name_to_fold: dict[str, int] = {}
                for fold_id, rep in enumerate(
                    sorted(collapsed, key=lambda r: (-len(collapsed[r]), r))
                ):
                    for member in collapsed[rep]:
                        name_to_fold[member] = fold_id

                counts = Counter(name_to_fold.values())
                sizes = [counts.get(i, 0) for i in range(k)]
                min_sz = min(sizes) if sizes else 0
                balanced = min_sz > 0 and max(sizes) / min_sz <= BALANCE_RATIO
                if verbose:
                    if min_sz == 0:
                        detail = "rejected (empty fold)"
                    elif balanced:
                        detail = f"accepted (ratio {max(sizes) / min_sz:.2f} ≤ {BALANCE_RATIO})"
                    else:
                        detail = f"rejected (ratio {max(sizes) / min_sz:.2f} > {BALANCE_RATIO})"
                    print(
                        f"[seqsplit] {stem}: id={seq_id:.2f} k={k} sizes={sizes} {detail}",
                        file=sys.stderr,
                    )
                if balanced:
                    out_csv = splits_dir / f"{stem}_seqid_folds.csv"
                    with csv_path.open(newline="", encoding="utf-8") as inf:
                        reader = csv.DictReader(inf)
                        fieldnames = list(reader.fieldnames or [])
                        rows = list(reader)
                    fieldnames.append("fold")
                    out_csv.parent.mkdir(parents=True, exist_ok=True)
                    with out_csv.open("w", newline="", encoding="utf-8") as outf:
                        writer = csv.DictWriter(
                            outf, fieldnames=fieldnames, extrasaction="ignore"
                        )
                        writer.writeheader()
                        for row in rows:
                            name = row[name_col].strip()
                            row["fold"] = str(name_to_fold[name])
                            writer.writerow(row)
                    if verbose:
                        print(
                            f"[seqsplit] {stem}: wrote {out_csv.name} "
                            f"(id={seq_id:.2f}, {k} folds, sizes={sizes})",
                            file=sys.stderr,
                        )
                    return {
                        "success": True,
                        "csv_path": out_csv,
                        "split_col": "fold",
                        "n_folds": k,
                        "seq_id": seq_id,
                    }

        return _fallback("no balanced split found across all id thresholds")
    finally:
        fasta_path.unlink(missing_ok=True)
