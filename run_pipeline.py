"""run_pipeline.py — the one-command orchestrator promised in the README.

Pipeline order (each stage's artifact is verified before the next one runs):

    generate   src.data_generation   -> data/raw_sales.csv
    process    src.data_processing   -> data/processed_sales.parquet
    analyze    src.analysis          -> insights/executive_memo.md
    visualize  src.visualize         -> insights/charts/

Stages communicate through files on disk, so any stage can be re-run on its
own and every intermediate output stays inspectable.

MODULE CONTRACT — every src module must expose::

    def main() -> Path:
        """Run this module's full step; return the primary artifact path."""

Modules are imported lazily, so the pipeline can be built incrementally.
``--dry-run`` reports which stages are ready without executing anything.

Usage:
    python run_pipeline.py                      # full deterministic rebuild
    python run_pipeline.py --skip-generate      # reuse the existing raw CSV
    python run_pipeline.py --stages analyze,visualize
    python run_pipeline.py --dry-run            # plan + readiness check only

Exit codes: 0 = success, 1 = stage/pipeline failure, 2 = usage error,
130 = interrupted (Ctrl+C).
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Make `src.*` importable regardless of the directory this is run from.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

try:  # scaffolding guard — src/config.py ships with the repo
    from src.config import get_config, setup_logging
except ImportError as exc:  # pragma: no cover
    sys.exit(f"FATAL: cannot import src/config.py ({exc}). It is part of the "
             f"repo scaffolding and must exist before running the pipeline.")


# --------------------------------------------------------------------------- #
# Stage registry
# --------------------------------------------------------------------------- #

class PipelineError(RuntimeError):
    """User-facing failure with an actionable message."""


class StageNotReady(PipelineError):
    """A src module is missing, unimportable, or lacks a main() function."""


@dataclass(frozen=True)
class Stage:
    key: str        # CLI name used by --stages
    module: str     # dotted import path
    file_hint: str  # where to create the module if it is missing
    label: str      # one-line human description

    def artifact_path(self, cfg: dict[str, Any]) -> Path:
        """File/dir this stage must produce (paths come from config.yaml)."""
        mapping = {
            "generate": cfg["paths"]["raw_data"],
            "process": cfg["paths"]["processed_data"],
            "analyze": cfg["paths"]["memo_file"],
            "visualize": cfg["paths"]["charts_dir"],
        }
        return Path(mapping[self.key])


STAGES: tuple[Stage, ...] = (
    Stage("generate", "src.data_generation", "src/data_generation.py",
          "Generate the synthetic raw sales dataset"),
    Stage("process", "src.data_processing", "src/data_processing.py",
          "Clean data + engineer business features"),
    Stage("analyze", "src.analysis", "src/analysis.py",
          "Compute KPIs, rank insights, write the executive memo"),
    Stage("visualize", "src.visualize", "src/visualize.py",
          "Export decision-oriented charts"),
)

#: stage key -> config path key of the input artifact it depends on
STAGE_INPUTS: dict[str, str] = {
    "process": "raw_data",
    "analyze": "processed_data",
    "visualize": "processed_data",
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "One-command orchestrator: generate -> process -> analyze -> "
            "visualize. Each stage calls main() in its src/ module and its "
            "output artifact is verified before the next stage runs."
        ),
    )
    parser.add_argument(
        "--stages",
        default="generate,process,analyze,visualize",
        help="comma-separated subset to run; order is normalised for you",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="reuse the existing raw CSV instead of regenerating (fast iteration)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the plan and each stage's readiness; execute nothing",
    )
    return parser.parse_args(argv)


def select_stages(spec: str) -> list[Stage]:
    """Parse a --stages value; canonical order is always preserved."""
    valid = [s.key for s in STAGES]
    requested = {p.strip().lower() for p in spec.split(",") if p.strip()}
    if not requested:
        raise PipelineError(
            f"--stages was empty. Valid stages: {', '.join(valid)}."
        )
    unknown = requested.difference(valid)
    if unknown:
        raise PipelineError(
            f"Unknown stage(s): {', '.join(sorted(unknown))}. "
            f"Valid stages: {', '.join(valid)}."
        )
    return [s for s in STAGES if s.key in requested]


# --------------------------------------------------------------------------- #
# Stage helpers
# --------------------------------------------------------------------------- #

def resolve_main(stage: Stage) -> Callable[[], Path]:
    """Import the stage's module lazily and return its main() function."""
    try:
        module = importlib.import_module(stage.module)
    except ImportError as exc:
        raise StageNotReady(
            f"'{stage.module}' could not be imported.\n"
            f"  Either it is not built yet (create {stage.file_hint} exposing "
            f"'main() -> Path')\n"
            f"  or it failed to import (missing dependency / syntax error) — "
            f"the full traceback is in the log file."
        ) from exc
    main = getattr(module, "main", None)
    if not callable(main):
        raise StageNotReady(
            f"'{stage.module}' exists but exposes no callable main().\n"
            f"  Add:  def main() -> Path: ..."
        )
    return main


def check_inputs(stage: Stage, cfg: dict[str, Any]) -> None:
    """Fail fast (with guidance) if a stage's input artifact is missing."""
    input_key = STAGE_INPUTS.get(stage.key)
    if input_key is None:
        return
    path = Path(cfg["paths"][input_key])
    if not path.is_file() or path.stat().st_size == 0:
        raise PipelineError(
            f"Stage '{stage.key}' needs {path}, which is missing or empty.\n"
            f"  Run the earlier stages first: python run_pipeline.py (full "
            f"run), or include them via --stages."
        )


def _fmt_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:,.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} GB"


def verify_artifact(stage: Stage, path: Path) -> str:
    """Confirm the stage really produced its artifact; return display text."""
    if path.is_dir():
        n_files = sum(1 for f in path.iterdir() if f.is_file())
        if n_files == 0:
            raise PipelineError(
                f"'{stage.key}' finished but {path} holds no files."
            )
        return f"{path} ({n_files} files)"
    if not path.is_file():
        raise PipelineError(
            f"'{stage.key}' finished but expected artifact {path} was not "
            f"created."
        )
    if path.stat().st_size == 0:
        raise PipelineError(
            f"'{stage.key}' finished but {path} is empty (0 bytes)."
        )
    return f"{path} ({_fmt_size(path.stat().st_size)})"


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def print_banner(cfg: dict[str, Any], stages: list[Stage]) -> None:
    g = cfg["generation"]
    print("=" * 68)
    print("  Data-to-Decision Accelerator — pipeline run")
    print(f"    seed={g['seed']} | orders={g['n_orders']:,} | "
          f"{g['start_date']} -> {g['end_date']}")
    print(f"    stages: {' -> '.join(s.key for s in stages)}")
    print("=" * 68 + "\n")


def dry_run(
    stages: list[Stage], cfg: dict[str, Any], logger: logging.Logger
) -> int:
    """Report the plan + each module's readiness without executing anything."""
    print("DRY RUN — nothing will be executed.\n")
    ready = 0
    for stage in stages:
        try:
            resolve_main(stage)
            status, detail = "READY ", stage.label
            ready += 1
        except StageNotReady as exc:
            status, detail = "MISSING", str(exc).splitlines()[0]
        artifact = stage.artifact_path(cfg)
        state = ("exists (will be overwritten)" if artifact.exists()
                 else "will be created")
        print(f"  [{status}] {stage.key:<10} {stage.module:<22} -> {artifact}")
        print(f"          {detail}")
        print(f"          artifact {state}\n")
    logger.info("Dry run: %d/%d stages ready", ready, len(stages))
    print(f"{ready}/{len(stages)} stages ready.")
    if ready < len(stages):
        print("Build the missing modules (each needs 'def main() -> Path'), "
              "then re-run.")
        return 1
    print("All good — run without --dry-run to execute.")
    return 0


def run(
    stages: list[Stage],
    cfg: dict[str, Any],
    skip_generate: bool,
    logger: logging.Logger,
) -> int:
    """Execute stages in order, verifying artifacts; return an exit code."""
    results: list[tuple[Stage, float, str]] = []
    started = time.perf_counter()

    for stage in stages:
        if skip_generate and stage.key == "generate":
            raw = Path(cfg["paths"]["raw_data"])
            if raw.is_file() and raw.stat().st_size > 0:
                print(f"  [SKIP] generate — reusing existing {raw}\n")
                continue
            raise PipelineError(
                f"--skip-generate was given but {raw} is missing or empty.\n"
                f"  Run once without the flag to create the raw data."
            )

        check_inputs(stage, cfg)

        try:
            stage_main = resolve_main(stage)
        except StageNotReady as exc:
            logger.exception("Stage '%s' is not ready.", stage.key)
            print(f"  [FAIL] {stage.key} is not ready:\n{exc}\n"
                  f"         full traceback: {cfg['paths']['log_file']}",
                  file=sys.stderr)
            return 1

        print(f"  [RUN ] {stage.key:<10} {stage.label}")
        logger.info("Stage '%s' started", stage.key)
        t0 = time.perf_counter()
        try:
            produced: Path | None = stage_main()
        except Exception:
            logger.exception("Stage '%s' raised an exception.", stage.key)
            print(f"  [FAIL] {stage.key} raised an exception — traceback "
                  f"logged to {cfg['paths']['log_file']}\n"
                  f"         quick look: tail -40 {cfg['paths']['log_file']}",
                  file=sys.stderr)
            return 1
        elapsed = time.perf_counter() - t0

        # Path() coerces a sloppy str return; `or` covers a forgotten return.
        artifact = Path(produced) if produced else stage.artifact_path(cfg)
        try:
            status = verify_artifact(stage, artifact)
        except PipelineError as exc:
            logger.error(
                "Artifact verification failed for '%s': %s", stage.key, exc
            )
            print(f"  [FAIL] {stage.key} — {exc}", file=sys.stderr)
            return 1

        logger.info("Stage '%s' finished in %.1fs -> %s",
                    stage.key, elapsed, status)
        results.append((stage, elapsed, status))
        print(f"  [ OK ] {stage.key:<10} {elapsed:6.1f}s  {status}\n")

    print_summary(results, time.perf_counter() - started, cfg)
    return 0


def print_summary(
    results: list[tuple[Stage, float, str]],
    elapsed_total: float,
    cfg: dict[str, Any],
) -> None:
    line = "=" * 68
    if not results:
        print("Nothing was executed — all selected stages were skipped.")
        return
    print(f"\n{line}")
    print(f" PIPELINE COMPLETE — {len(results)} stage(s) in "
          f"{elapsed_total:.1f}s")
    print(line)
    for stage, elapsed, status in results:
        print(f"  {stage.key:<10} {elapsed:>6.1f}s   {status}")
    print(line)
    print(f"  Executive memo : {cfg['paths']['memo_file']}")
    print(f"  Charts folder  : {cfg['paths']['charts_dir']}/")
    print(f"  Full log       : {cfg['paths']['log_file']}")
    print("\n  Next step: streamlit run app.py\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        cfg = get_config()
    except FileNotFoundError as exc:
        print(f"FATAL: config.yaml not found at {exc.filename} — it must sit "
              f"at the repo root.", file=sys.stderr)
        return 1

    logger = setup_logging()

    try:
        stages = select_stages(args.stages)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        return dry_run(stages, cfg, logger)

    # Create output folders up front so stage errors are never "no folder".
    for key in ("raw_data", "processed_data", "memo_file", "charts_dir"):
        Path(cfg["paths"][key]).parent.mkdir(parents=True, exist_ok=True)

    print_banner(cfg, stages)
    logger.info("Pipeline started | stages=%s | skip_generate=%s",
                [s.key for s in stages], args.skip_generate)

    try:
        return run(stages, cfg, args.skip_generate, logger)
    except PipelineError as exc:
        logger.error("Pipeline aborted: %s", exc)
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C).")
        print("\nInterrupted — re-run to rebuild any partial artifacts.",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
