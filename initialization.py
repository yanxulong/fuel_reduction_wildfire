"""Path registry and ordered task runner for fuel-reduction wildfire work."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import tasks


TASK_REGISTRY: dict[str, Callable] = {
    "build_wui_singlepart_dataset": tasks.build_wui_singlepart_dataset,
    "build_fire_wui_geometry_characteristics": tasks.build_fire_wui_geometry_characteristics,
    "build_fire_wui_centroid_paths": tasks.build_fire_wui_centroid_paths,
    "build_fire_wui_centroid_path_distance_samples": tasks.build_fire_wui_centroid_path_distance_samples,
    "build_wui_annual_mtbs_outcomes": tasks.build_wui_annual_mtbs_outcomes,
    "landfire_controls": tasks.landfire_controls,
    "prepare_facts_treatments": tasks.prepare_facts_treatments,
    "build_fire_wui_treatment_definitions": tasks.build_fire_wui_treatment_definitions,
    "finalize_fire_wui_treatment_definitions": tasks.finalize_fire_wui_treatment_definitions,
}


class LocalProjectFlow:
    """Small ProjectFlow runner with explicit task order and shared paths."""

    def __init__(self) -> None:
        self.tasks: list[tuple[str, Callable]] = []
        self.paths: dict[str, Path] = {}
        self.config: dict[str, object] = {}
        self.results: dict[str, object] = {}

    def add_task(self, name: str, task_function: Callable) -> None:
        self.tasks.append((name, task_function))

    def execute(self) -> dict[str, object]:
        total = len(self.tasks)
        for index, (name, task_function) in enumerate(self.tasks, start=1):
            print(f"[task {index}/{total}] START {name}")
            try:
                result = task_function(self)
            except Exception as exc:
                print(f"[task {index}/{total}] FAILED {name}: {exc}")
                raise
            self.results[name] = result
            print(f"[task {index}/{total}] DONE {name}")
        return self.results


def initialize_paths(project: LocalProjectFlow, config: dict[str, object] | None = None) -> LocalProjectFlow:
    """Resolve every project location once and expose paths through project.paths."""
    config = dict(config or {})
    code_root = Path(__file__).resolve().parent
    checkout_root = code_root.parent
    home_root = Path.home() / "projects" / "fuel_reduction_wildfire"
    scratch_root = Path("/scratch.global/yxlong/projects/fuel_reduction_wildfire")
    configured_root = os.environ.get("FUEL_REDUCTION_PROJECT_ROOT") or config.get("project_root")
    if configured_root:
        project_root = Path(configured_root).expanduser().resolve()
    else:
        project_root = (scratch_root if scratch_root.is_dir() else checkout_root).resolve()

    extra_roots = [
        Path(value).expanduser()
        for value in os.environ.get("FUEL_REDUCTION_DATA_ROOTS", "").split(os.pathsep)
        if value.strip()
    ]

    def unique_paths(paths: list[Path]) -> tuple[Path, ...]:
        unique: list[Path] = []
        for path in paths:
            resolved = Path(path).expanduser().resolve()
            if resolved not in unique:
                unique.append(resolved)
        return tuple(unique)

    # Inputs may live across Home, Scratch, and the active local checkout.
    data_roots = unique_paths([home_root, scratch_root, *extra_roots, checkout_root, project_root])
    output_roots = unique_paths([project_root, home_root, scratch_root, *extra_roots, checkout_root])

    def nonempty_file(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    def nonempty_tree(path: Path, pattern: str = "*") -> bool:
        return path.is_dir() and any(
            candidate.is_file() and candidate.stat().st_size > 0
            for candidate in path.rglob(pattern)
        )

    def candidate_files(roots: tuple[Path, ...], relative_path: Path) -> tuple[Path, ...]:
        return tuple(root / relative_path for root in roots)

    def spatial_candidates(subdir: str, filename: str) -> tuple[Path, ...]:
        found: list[Path] = []
        for root in data_roots:
            for base in (
                root / "data" / "raw" / subdir,
                root / "data" / "archive" / "raw" / subdir,
            ):
                if base.is_dir():
                    direct = base / filename
                    if direct.is_file() and direct.stat().st_size > 0:
                        found.append(direct.resolve())
                    found.extend(
                        path.resolve()
                        for path in sorted(base.rglob(filename))
                        if path.is_file() and path.stat().st_size > 0
                    )
        return tuple(dict.fromkeys(found))

    def existing_product(
        dataset_relative: Path,
        summary_names: tuple[str, ...],
        *,
        pattern: str = "*",
    ) -> tuple[Path, tuple[Path, ...]] | None:
        for root in output_roots:
            dataset = root / dataset_relative
            summaries = tuple(root / "project" / "output" / "tables" / name for name in summary_names)
            if nonempty_tree(dataset, pattern) and all(nonempty_file(path) for path in summaries):
                return dataset, summaries
        return None

    def first_existing_file(candidates: tuple[Path, ...], fallback: Path) -> Path:
        return next((path.resolve() for path in candidates if nonempty_file(path)), fallback)

    project_dir = project_root / "project"
    output_dir = project_dir / "output"

    wui_paths = {
        year: first_existing_file(
            candidate_files(data_roots, Path("data/processed/wui_by_year") / f"silvis_wui_{year}.fgb"),
            project_root / "data" / "processed" / "wui_by_year" / f"silvis_wui_{year}.fgb",
        )
        for year in (2000, 2010, 2020)
    }
    fod_candidates = spatial_candidates("MTBS", "mtbs_FODpoints_DD.shp")
    perimeter_candidates = spatial_candidates("MTBS", "mtbs_perims_DD.shp")
    facts_candidates = spatial_candidates("FACTS", "Actv_HazFuelTrt_PL.shp")

    landfire_roots = unique_paths([
        scratch_root,
        home_root,
        *extra_roots,
        checkout_root,
        project_root,
    ])
    landfire_root_candidates = unique_paths([
        root / "data" / "raw" / "science_controls" / "LANDFIRE" / "LF2020"
        for root in landfire_roots
    ])
    landfire_root = next(
        (
            candidate
            for candidate in landfire_root_candidates
            if all((candidate / layer).is_dir() for layer in ("Elev", "Asp", "SlpD"))
        ),
        landfire_root_candidates[0],
    )

    mtbs_raster_roots = unique_paths([
        root / relative
        for root in data_roots
        for relative in (
            Path("data/raw/MTBS/MTBS_BSmosaics"),
            Path("data/archive/raw/MTBS/MTBS_BSmosaics"),
        )
    ])

    wui_product = existing_product(
        Path("data/processed/wui_singlepart"),
        ("wui_singlepart_summary.csv",),
        pattern="*.parquet",
    )
    wui_singlepart_dataset, wui_summary_paths = wui_product or (
        project_root / "data" / "processed" / "wui_singlepart",
        (output_dir / "tables" / "wui_singlepart_summary.csv",),
    )

    outcomes_product = existing_product(
        Path("data/processed/wui_annual_mtbs_outcomes"),
        ("wui_annual_mtbs_outcomes_summary.csv",),
        pattern="*.parquet",
    )
    outcomes_dataset, outcomes_summary_paths = outcomes_product or (
        project_root / "data" / "processed" / "wui_annual_mtbs_outcomes",
        (output_dir / "tables" / "wui_annual_mtbs_outcomes_summary.csv",),
    )

    treatment_product = existing_product(
        Path("data/processed/fire_wui_treatment_definitions"),
        ("fire_wui_treatment_definitions_summary.csv", "treatment_definition_catalog.csv"),
    )
    treatment_dataset, treatment_summary_paths = treatment_product or (
        project_root / "data" / "processed" / "fire_wui_treatment_definitions",
        (
            output_dir / "tables" / "fire_wui_treatment_definitions_summary.csv",
            output_dir / "tables" / "treatment_definition_catalog.csv",
        ),
    )

    path_distances = tuple(round(1.0 + 0.5 * index, 1) for index in range(19))
    path_root_scores: list[tuple[int, Path]] = []
    for root in output_roots:
        dataset_root = root / "data" / "processed" / "fire_wui_centroid_paths"
        tables_root = root / "project" / "output" / "tables"
        completed = sum(
            nonempty_tree(dataset_root / f"distance_{distance:g}mi")
            and nonempty_file(tables_root / f"fire_wui_centroid_paths_{distance:g}mi_summary.csv")
            for distance in path_distances
        )
        path_root_scores.append((completed, root))
    best_path_score, best_path_root = max(path_root_scores, key=lambda item: item[0])
    path_root = best_path_root if best_path_score else project_root

    facts_treatment_path = next(
        (
            candidate.resolve()
            for root in output_roots
            for candidate in (root / "data" / "processed" / "facts_treatments_eligible.parquet",)
            if nonempty_file(candidate)
        ),
        project_root / "data" / "processed" / "facts_treatments_eligible.parquet",
    )

    raw_search_roots = unique_paths([
        root / relative
        for root in data_roots
        for relative in (
            Path("data/raw/MTBS"),
            Path("data/archive/raw/MTBS"),
        )
    ])
    facts_search_roots = unique_paths([
        root / relative
        for root in data_roots
        for relative in (
            Path("data/raw/FACTS"),
            Path("data/archive/raw/FACTS"),
        )
    ])

    print(f"[paths] project output root: {project_root}")
    print("[paths] input roots: " + " -> ".join(map(str, data_roots)))
    if best_path_score:
        print(f"[paths] reusing fire-path outputs at {path_root} ({best_path_score}/19 distances complete)")

    project.paths = {
        "code": code_root,
        "project_root": project_root,
        "data_roots": data_roots,
        "data": project_root / "data",
        "raw": project_root / "data" / "raw",
        "processed": project_root / "data" / "processed",
        "wui_by_year_2000": wui_paths[2000],
        "wui_by_year_2010": wui_paths[2010],
        "wui_by_year_2020": wui_paths[2020],
        "wui_singlepart_dataset": wui_singlepart_dataset,
        "fire_wui_geometry_characteristics_dataset": (
            project_root / "data" / "processed" / "fire_wui_geometry_characteristics"
        ),
        "fire_wui_geometry_characteristics_summary": (
            output_dir / "tables" / "fire_wui_geometry_characteristics_summary.csv"
        ),
        "fire_wui_path_dataset": path_root / "data" / "processed" / "fire_wui_centroid_paths",
        "fire_wui_path_summary": path_root / "project" / "output" / "tables" / "fire_wui_centroid_paths_summary.csv",
        "fire_wui_paths_10mi": path_root / "data" / "processed" / "fire_wui_centroid_paths" / "distance_10mi",
        "mtbs_fod_points_candidates": fod_candidates,
        "mtbs_perimeter_candidates": perimeter_candidates,
        "facts_treatment_candidates": facts_candidates,
        "landfire_raster_root": landfire_root,
        "landfire_elev_tiles": landfire_root / "Elev",
        "landfire_asp_tiles": landfire_root / "Asp",
        "landfire_slpd_tiles": landfire_root / "SlpD",
        "landfire_control_dataset": project_root / "data" / "processed" / "landfire_controls",
        "landfire_controls_summary": output_dir / "tables" / "landfire_controls_summary.csv",
        "mtbs_search_roots": raw_search_roots,
        "facts_search_roots": facts_search_roots,
        "facts_treatments_eligible": facts_treatment_path,
        "fire_wui_treatment_dataset": treatment_dataset,
        "fire_wui_treatment_summary": treatment_summary_paths[0],
        "treatment_definition_catalog": treatment_summary_paths[1],
        "fire_wui_treatment_runs": treatment_dataset.parent / ".fire_wui_treatment_runs",
        "mtbs_severity_raster_roots": mtbs_raster_roots,
        "mtbs_severity_raster_root": mtbs_raster_roots[0],
        "wui_annual_mtbs_outcomes": outcomes_dataset,
        "wui_annual_mtbs_outcomes_summary": outcomes_summary_paths[0],
        "data_archive": project_root / "data" / "archive",
        "project": project_dir,
        "intermediate": project_dir / "intermediate",
        "output": output_dir,
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "wui_singlepart_summary": wui_summary_paths[0],
        "report": output_dir / "report",
        "presentation": output_dir / "presentation",
        "logs": output_dir / "logs",
    }
    project.config = config
    project.results = {}
    return project


def build_task_tree(project: LocalProjectFlow, config: dict[str, object]) -> LocalProjectFlow:
    """Validate configured task names and register tasks in requested order."""
    enabled_tasks = list(config.get("enabled_tasks", []))
    unknown_tasks = [name for name in enabled_tasks if name not in TASK_REGISTRY]
    if unknown_tasks:
        raise ValueError(f"Unknown enabled tasks: {unknown_tasks}")
    project.tasks.clear()
    for task_name in enabled_tasks:
        project.add_task(task_name, TASK_REGISTRY[task_name])
    return project
