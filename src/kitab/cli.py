"""kitAb command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _repo_root_from_here() -> Path:
    # src/kitab/cli.py -> repo root
    return Path(__file__).resolve().parents[2]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kitab",
        description="kitAb: antibody developability descriptors + AutoML",
    )
    sub = p.add_subparsers(dest="command", required=True)

    val_p = sub.add_parser(
        "validate",
        help="Check that a run YAML is valid (no compute)",
    )
    val_p.add_argument("config", type=Path, help="Run YAML path")

    run_p = sub.add_parser("run", help="Execute a kitAb pipeline")
    run_p.add_argument("config", type=Path, help="Run YAML path")
    _add_common_overrides(run_p)

    resume_p = sub.add_parser("resume", help="Resume an existing run directory")
    resume_p.add_argument("run_dir", type=Path)
    resume_p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional original config (default: <run_dir>/resolved_manifest.yaml source)",
    )
    return p


def _add_common_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--stages", type=str, default=None, help="Comma-separated stage list")
    p.add_argument("--cpus", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--enable-automl", action="store_true", default=None)
    p.add_argument("--disable-automl", action="store_true", default=None)
    p.add_argument(
        "--techniques",
        type=str,
        default=None,
        help="Comma-separated AutoML techniques (default: all four)",
    )
    p.add_argument(
        "--cv-mode",
        choices=("nested", "flat"),
        default=None,
        help="AutoML cross-validation mode (default: nested)",
    )
    p.add_argument(
        "--no-final-model",
        action="store_true",
        default=None,
        help="Compare techniques without refitting the winner on all data",
    )
    p.add_argument("--resume", action="store_true", default=None)


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if getattr(args, "output_dir", None):
        out["output_dir"] = args.output_dir
    if getattr(args, "stages", None):
        out["stages"] = [s.strip() for s in args.stages.split(",") if s.strip()]
    if getattr(args, "cpus", None) is not None:
        out["n_cpu"] = args.cpus
    if getattr(args, "device", None):
        out["device"] = args.device
    if getattr(args, "enable_automl", None):
        out["enable_automl"] = True
    if getattr(args, "disable_automl", None):
        out["enable_automl"] = False
    if getattr(args, "techniques", None):
        out["techniques"] = args.techniques
    if getattr(args, "cv_mode", None):
        out["cv_mode"] = args.cv_mode
    if getattr(args, "no_final_model", None):
        out["no_final_model"] = True
    if getattr(args, "resume", None):
        out["resume"] = True
    return out


def _print_config_error(exc: BaseException, path: Path | None) -> None:
    loc = f" in {path}" if path else ""
    issues = getattr(exc, "issues", None)
    if not isinstance(issues, list) or not issues:
        print(f"Config error{loc}: {exc}", file=sys.stderr)
        return
    if len(issues) == 1:
        print(f"Config error{loc}: {issues[0]}", file=sys.stderr)
        return
    print(f"Config error{loc}: {len(issues)} problems found", file=sys.stderr)
    for issue in issues:
        print(f"  - {issue}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = _repo_root_from_here()

    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from kitab.config import ConfigError, load_manifest
    from kitab.pipeline import run_pipeline

    cfg_path: Path | None = getattr(args, "config", None)
    try:
        if args.command == "resume":
            run_dir = args.run_dir.resolve()
            resolved = run_dir / "resolved_manifest.yaml"
            if args.config is not None:
                cfg_path = args.config
            elif resolved.is_file():
                from kitab.config import load_yaml

                data = load_yaml(resolved)
                src_path = data.get("source_path")
                cfg_path = Path(src_path) if src_path else resolved
            else:
                raise SystemExit(
                    f"Cannot resume: missing {resolved} and no --config provided"
                )
            overrides = _overrides_from_args(args)
            overrides["resume"] = True
            overrides["output_dir"] = str(run_dir)
            manifest = load_manifest(cfg_path, repo_root=repo_root, cli_overrides=overrides)
            return run_pipeline(manifest)

        if args.command == "validate":
            load_manifest(args.config, repo_root=repo_root)
            print(f"OK: {args.config}")
            return 0

        manifest = load_manifest(
            args.config, repo_root=repo_root, cli_overrides=_overrides_from_args(args)
        )
        if args.command == "run":
            return run_pipeline(manifest)
        parser.error(f"Unknown command {args.command}")
        return 2
    except ConfigError as exc:
        _print_config_error(exc, cfg_path)
        return 2
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
