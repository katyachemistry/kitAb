#!/usr/bin/env python3
"""ABB2 / ABB3 batch structure prediction from CSV (name, heavy, light)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Union

ABB3_SRC_DEFAULT = os.environ.get("ABB3_SRC", "/home/kb/abodybuilder3/src")
CKPT_DEFAULT = os.environ.get(
    "ABB3_CHECKPOINT",
    "/home/kb/abodybuilder3/output/plddt-loss/best_second_stage.ckpt",
)


def _build_dataset_jobs(
    csv_paths: list[Path] | None,
    data_dir: Path | None,
    dataset_names: list[str] | None,
) -> list[tuple[Path, str]]:
    if data_dir is not None:
        if csv_paths:
            raise SystemExit("Use either --data-dir or --csv, not both")
        d = data_dir.resolve()
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")
        paths = sorted(d.glob("*.csv"))
        if not paths:
            raise SystemExit(f"No *.csv files in {d}")
        if dataset_names:
            if len(dataset_names) != len(paths):
                raise SystemExit(
                    f"--dataset count ({len(dataset_names)}) must match "
                    f"CSV count ({len(paths)}) in --data-dir, or omit --dataset for stems"
                )
            return list(zip(paths, dataset_names))
        return [(p, p.stem) for p in paths]

    if not csv_paths:
        raise SystemExit("Provide --csv (one or more) or --data-dir")

    names = dataset_names or []
    if not names:
        return [(p.resolve(), p.stem) for p in csv_paths]
    if len(names) == len(csv_paths):
        return [(p.resolve(), n) for p, n in zip(csv_paths, names)]
    raise SystemExit(
        "With --csv: give one --dataset per --csv (same order), or omit --dataset "
        "to use each file's basename (without .csv)"
    )


def _validate_csv_columns(csv_path: Path) -> None:
    import pandas as pd

    header = pd.read_csv(csv_path, nrows=0)
    missing = [c for c in ("name", "heavy", "light") if c not in header.columns]
    if missing:
        raise SystemExit(f"{csv_path}: missing columns {missing}")


def _early_set_cuda_visible_device(device_arg: str) -> None:
    dev = device_arg.strip()
    if not dev:
        dev = os.environ.get("ABB2_DEVICE", "").strip()
    if not dev:
        dev = "cuda:1"
    if dev == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    if dev.startswith("cuda:"):
        tail = dev.split(":", 1)[1].strip()
        if tail.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = tail
            return
    if dev == "cuda":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return
    raise SystemExit(f"Unknown --device {dev!r} (use cuda:0, cuda:1, cpu)")


def _resolve_torch_device(device_arg: str):
    import torch

    device_arg = device_arg.strip()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(f"Device {device_arg} requested but CUDA is not available")
        device = torch.device(device_arg)
        idx = device.index if device.index is not None else 0
        if idx >= torch.cuda.device_count():
            raise SystemExit(
                f"Device {device_arg} not available "
                f"(only {torch.cuda.device_count()} CUDA device(s))"
            )
        return device
    raise SystemExit(f"Unknown --device {device_arg!r} (use cuda:0, cuda:1, cpu)")


def process_one_dataset_abb2(
    csv_path: Path,
    dataset: str,
    predictor,
    out_root: Path,
    runs: int,
    skip_existing: bool,
    n_threads_save: int,
) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = [c for c in ("name", "heavy", "light") if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path}: missing columns {missing}")

    print(f"\n>>> Dataset {dataset!r} ({csv_path.name})", flush=True)

    for run in range(1, runs + 1):
        run_dir = out_root / f"{dataset}_abb2_{run}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== run {run}/{runs} -> {run_dir} ===", flush=True)

        n_ok, n_skip, n_fail = 0, 0, 0

        for _, row in df.iterrows():
            name = str(row["name"]).strip() if pd.notna(row["name"]) else ""
            heavy = str(row["heavy"]).strip() if pd.notna(row["heavy"]) else ""
            light = str(row["light"]).strip() if pd.notna(row["light"]) else ""
            if not name or not heavy or not light:
                n_fail += 1
                continue

            pdb_path = run_dir / f"{name}.pdb"
            if skip_existing and pdb_path.is_file():
                n_skip += 1
                continue

            sequences = {"H": heavy, "L": light}
            try:
                antibody = predictor.predict(sequences)
            except Exception as e:
                print(f"  FAIL {name!r}: {e}", flush=True)
                n_fail += 1
                continue

            if antibody is None:
                n_fail += 1
                continue

            try:
                antibody.save(
                    str(pdb_path),
                    check_for_strained_bonds=True,
                    n_threads=n_threads_save,
                )
                n_ok += 1
            except Exception as e:
                print(f"  FAIL {name!r} (save): {e}", flush=True)
                n_fail += 1

        print(
            f"  run {run}: wrote {n_ok}, skipped {n_skip}, failed {n_fail}",
            flush=True,
        )


def main_abb2(args: argparse.Namespace) -> None:
    _early_set_cuda_visible_device(args.device)
    import torch
    from ImmuneBuilder import ABodyBuilder2

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    csv_arg_list = [Path(x) for x in args.csv] if args.csv else None
    dataset_names = list(args.dataset) if args.dataset else None
    jobs = _build_dataset_jobs(csv_arg_list, args.data_dir, dataset_names)

    for csv_path, _stem in jobs:
        if not csv_path.is_file():
            raise SystemExit(f"Not a file: {csv_path}")
        _validate_csv_columns(csv_path)

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}", flush=True)
    predictor = ABodyBuilder2()
    print(f"ABodyBuilder2 device: {predictor.device}", flush=True)

    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Datasets to run ({len(jobs)}):", flush=True)
    for csv_path, stem in jobs:
        print(f"  - {stem!r} <- {csv_path}", flush=True)

    n_threads_save = args.refine_threads
    if n_threads_save == 0:
        n_threads_save = -1

    for csv_path, dataset in jobs:
        process_one_dataset_abb2(
            csv_path,
            dataset,
            predictor,
            out_root,
            args.runs,
            args.skip_existing,
            n_threads_save,
        )

    print(f"\nDone. Outputs under {out_root}", flush=True)


def strings_to_datapoint(heavy: str, light: str) -> dict:
    import torch
    from abodybuilder3.dataloader import ABDataset
    from abodybuilder3.openfold.np.residue_constants import restype_order_with_x

    aatype: list[int] = []
    is_heavy: list[int] = []
    for c in heavy:
        is_heavy.append(1)
        aatype.append(restype_order_with_x[c])
    for c in light:
        is_heavy.append(0)
        aatype.append(restype_order_with_x[c])
    is_heavy_t = torch.tensor(is_heavy, dtype=torch.long)
    aatype_t = torch.tensor(aatype, dtype=torch.long)
    residue_index = torch.cat(
        (torch.arange(len(heavy)), torch.arange(len(light)) + 500)
    )
    model_input = {
        "is_heavy": is_heavy_t,
        "aatype": aatype_t,
        "residue_index": residue_index,
    }
    model_input.update(
        ABDataset.single_and_double_from_datapoint(
            model_input, 64, edge_chain_feature=True
        )
    )
    n = model_input["aatype"].shape[0]
    model_input["seq_mask"] = torch.ones(n, dtype=torch.float32)
    return model_input


def collate_padded(samples: list[dict], pad_aatype: int) -> dict:
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from abodybuilder3.dataloader import pad_square_tensors

    out: dict = {}
    out["pair"] = pad_square_tensors([s["pair"] for s in samples])
    out["aatype"] = pad_sequence(
        [s["aatype"] for s in samples],
        batch_first=True,
        padding_value=pad_aatype,
    )
    out["is_heavy"] = pad_sequence(
        [s["is_heavy"] for s in samples], batch_first=True, padding_value=0
    )
    out["residue_index"] = pad_sequence(
        [s["residue_index"] for s in samples], batch_first=True, padding_value=0
    )
    out["single"] = pad_sequence(
        [s["single"].float() for s in samples], batch_first=True, padding_value=0.0
    )
    out["seq_mask"] = pad_sequence(
        [s["seq_mask"].float() for s in samples], batch_first=True, padding_value=0.0
    )
    return out


def pdb_from_batched_output(
    full_output: dict,
    batch: dict,
    batch_idx: int,
    n_res: int,
    datapoint_cpu: dict,
) -> str:
    from abodybuilder3.utils import add_atom37_to_output, output_to_pdb

    sub_out = {"positions": full_output["positions"][:, batch_idx : batch_idx + 1, :n_res]}
    aatype_1d = batch["aatype"][batch_idx, :n_res]
    add_atom37_to_output(sub_out, aatype_1d)
    sub_out["atom37"] = sub_out["atom37"].detach().cpu()
    sub_out["atom37_atom_exists"] = sub_out["atom37_atom_exists"].detach().cpu()
    mi = {
        "aatype": datapoint_cpu["aatype"][:n_res].clone(),
        "is_heavy": datapoint_cpu["is_heavy"][:n_res].clone(),
    }
    return output_to_pdb(sub_out, mi)


def run_batch_inference(
    model,
    device,
    rows: list[tuple[str, dict]],
    pad_aatype: int,
) -> dict[str, Union[str, Exception]]:
    import torch

    if not rows:
        return {}
    batch_cpu = collate_padded([d for _, d in rows], pad_aatype)
    batch = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_cpu.items()
    }
    evo = {"single": batch["single"], "pair": batch["pair"]}
    with torch.inference_mode():
        full_out = model(evo, batch["aatype"], batch["seq_mask"])
    out: dict[str, Union[str, Exception]] = {}
    for b, (name, dp_cpu) in enumerate(rows):
        n_res = int(batch["seq_mask"][b].sum().item())
        try:
            out[name] = pdb_from_batched_output(full_out, batch, b, n_res, dp_cpu)
        except Exception as e:
            out[name] = e
    return out


def process_one_dataset_abb3(
    csv_path: Path,
    dataset: str,
    model,
    device,
    pad_aatype: int,
    out_root: Path,
    runs: int,
    batch_size: int,
    skip_existing: bool,
) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = [c for c in ("name", "heavy", "light") if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path}: missing columns {missing}")

    print(f"\n>>> Dataset {dataset!r} ({csv_path.name})", flush=True)

    for run in range(1, runs + 1):
        run_dir = out_root / f"{dataset}_abb3_{run}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== run {run}/{runs} -> {run_dir} ===", flush=True)

        n_ok, n_skip, n_fail = 0, 0, 0
        buf: list[tuple[str, str, str, dict]] = []

        def flush_buffer() -> None:
            nonlocal n_ok, n_fail, buf
            if not buf:
                return
            rows = [(name, dp) for name, _, _, dp in buf]
            names_in_batch = [name for name, _, _, _ in buf]
            results = run_batch_inference(model, device, rows, pad_aatype)
            for name in names_in_batch:
                pdb_path = run_dir / f"{name}.pdb"
                r = results[name]
                if isinstance(r, Exception):
                    print(f"  FAIL {name!r}: {r}", flush=True)
                    n_fail += 1
                else:
                    pdb_path.write_text(r, encoding="utf-8")
                    n_ok += 1
            buf.clear()

        for _, row in df.iterrows():
            name = str(row["name"]).strip() if pd.notna(row["name"]) else ""
            heavy = str(row["heavy"]).strip() if pd.notna(row["heavy"]) else ""
            light = str(row["light"]).strip() if pd.notna(row["light"]) else ""
            if not name or not heavy or not light:
                n_fail += 1
                continue

            pdb_path = run_dir / f"{name}.pdb"
            if skip_existing and pdb_path.is_file():
                n_skip += 1
                continue

            try:
                dp = strings_to_datapoint(heavy, light)
            except Exception as e:
                print(f"  FAIL {name!r} (bad sequence?): {e}", flush=True)
                n_fail += 1
                continue

            buf.append((name, heavy, light, dp))
            if len(buf) >= batch_size:
                flush_buffer()

        flush_buffer()

        print(
            f"  run {run}: wrote {n_ok}, skipped {n_skip}, failed {n_fail}",
            flush=True,
        )


def main_abb3(args: argparse.Namespace) -> None:
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    csv_arg_list = [Path(x) for x in args.csv] if args.csv else None
    dataset_names = list(args.dataset) if args.dataset else None
    jobs = _build_dataset_jobs(csv_arg_list, args.data_dir, dataset_names)

    for csv_path, _stem in jobs:
        if not csv_path.is_file():
            raise SystemExit(f"Not a file: {csv_path}")
        _validate_csv_columns(csv_path)

    abb3_src = Path(ABB3_SRC_DEFAULT).resolve()
    if not abb3_src.is_dir():
        raise SystemExit(f"ABB3 source directory not found: {abb3_src} (set ABB3_SRC)")
    sys.path.insert(0, str(abb3_src))

    from abodybuilder3.lightning_module import LitABB3
    from abodybuilder3.openfold.np.residue_constants import restype_order_with_x

    pad_aatype = int(restype_order_with_x["X"])
    device = _resolve_torch_device(args.device)
    print(f"Device: {device}", flush=True)

    print(f"Loading checkpoint: {ckpt}", flush=True)
    module = LitABB3.load_from_checkpoint(str(ckpt), map_location=str(device))
    model = module.model
    model.eval()
    model.to(device)

    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Datasets to run ({len(jobs)}):", flush=True)
    for csv_path, stem in jobs:
        print(f"  - {stem!r} <- {csv_path}", flush=True)

    for csv_path, dataset in jobs:
        process_one_dataset_abb3(
            csv_path,
            dataset,
            model,
            device,
            pad_aatype,
            out_root,
            args.runs,
            args.batch_size,
            args.skip_existing,
        )

    print(f"\nDone. Outputs under {out_root}", flush=True)


def _add_shared_csv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--csv",
        action="append",
        default=None,
        metavar="PATH",
        help="Input CSV (repeat for multiple datasets).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Process every top-level *.csv in DIR (non-recursive).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="STEM",
        help="Output folder prefix per CSV (same order as --csv or sorted *.csv).",
    )
    parser.add_argument("--output-root", required=True, type=Path, help="structures/ parent")
    parser.add_argument("--runs", type=int, default=1, help="Run folders: _abb2_1 / _abb3_1 …")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip antibodies whose .pdb already exists in that run directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Mini-batch size (ABB3 only; accepted on ABB2 for CLI parity).",
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        help='Torch / CUDA device hint (default cuda:1). Use cpu to force CPU.',
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="backend", required=True)

    abb2_p = sub.add_parser(
        "abb2",
        help="AbodyBuilder2 (ImmuneBuilder); one sequence at a time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_shared_csv_args(abb2_p)
    abb2_p.add_argument(
        "--refine-threads",
        type=int,
        default=1,
        help="n_threads passed to OpenMM refinement in antibody.save (default: 1).",
    )

    abb3_p = sub.add_parser(
        "abb3",
        help="AbodyBuilder3 padded mini-batch inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_shared_csv_args(abb3_p)
    abb3_p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(CKPT_DEFAULT),
        help="LitABB3 checkpoint (default or ABB3_CHECKPOINT env)",
    )

    args = parser.parse_args(argv)
    if args.backend == "abb2":
        main_abb2(args)
    else:
        main_abb3(args)


if __name__ == "__main__":
    main()
