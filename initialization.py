"""Path registry and ordered task runner for fuel-reduction wildfire work."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import tasks


TASK_REGISTRY: dict[str, Callable] = {
    "build_wui_singlepart_dataset": tasks.build_wui_singlepart_dataset,
    "build_fire_wui_centroid_paths": tasks.build_fire_wui_centroid_paths,
    "build_fire_wui_centroid_path_distance_samples": tasks.build_fire_wui_centroid_path_distance_samples,
    "build_wui_annual_mtbs_outcomes": tasks.build_wui_annual_mtbs_outcomes,
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
    configured_root = os.environ.get("FUEL_REDUCTION_PROJECT_ROOT") or config.get("project_root")
    project_root = Path(configured_root or code_root.parent).expanduser().resolve()
    project_dir = project_root / "project"
    output_dir = project_dir / "output"

    project.paths = {
        "code": code_root,
        "project_root": project_root,
        "data": project_root / "data",
        "raw": project_root / "data" / "raw",
        "processed": project_root / "data" / "processed",
        "wui_by_year_2000": project_root / "data" / "processed" / "wui_by_year" / "silvis_wui_2000.fgb",
        "wui_by_year_2010": project_root / "data" / "processed" / "wui_by_year" / "silvis_wui_2010.fgb",
        "wui_by_year_2020": project_root / "data" / "processed" / "wui_by_year" / "silvis_wui_2020.fgb",
        "wui_singlepart_dataset": project_root / "data" / "processed" / "wui_singlepart",
        "fire_wui_path_dataset": project_root / "data" / "processed" / "fire_wui_centroid_paths",
        "fire_wui_path_summary": output_dir / "tables" / "fire_wui_centroid_paths_summary.csv",
        "mtbs_fod_points_candidates": (
            project_root / "data" / "raw" / "MTBS" / "mtbs_fod_pts_data" / "mtbs_FODpoints_DD.shp",
            project_root / "data" / "raw" / "MTBS" / "mtbs_FODpoints_DD.shp",
            project_root / "data" / "archive" / "raw" / "MTBS" / "mtbs_fod_pts_data" / "mtbs_FODpoints_DD.shp",
            project_root / "data" / "archive" / "raw" / "MTBS" / "mtbs_FODpoints_DD.shp",
        ),
        "mtbs_perimeter_candidates": (
            project_root / "data" / "raw" / "MTBS" / "mtbs_perims_DD.shp",
            project_root / "data" / "raw" / "MTBS" / "MTBS_Burn_Area" / "mtbs_perims_DD.shp",
            project_root / "data" / "archive" / "raw" / "MTBS" / "mtbs_perims_DD.shp",
            project_root / "data" / "archive" / "raw" / "MTBS" / "MTBS_Burn_Area" / "mtbs_perims_DD.shp",
        ),
        "mtbs_severity_raster_root": project_root / "data" / "raw" / "MTBS" / "MTBS_BSmosaics",
        "wui_annual_mtbs_outcomes": project_root / "data" / "processed" / "wui_annual_mtbs_outcomes",
        "wui_annual_mtbs_outcomes_summary": output_dir / "tables" / "wui_annual_mtbs_outcomes_summary.csv",
        "data_archive": project_root / "data" / "archive",
        "project": project_dir,
        "intermediate": project_dir / "intermediate",
        "output": output_dir,
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "wui_singlepart_summary": output_dir / "tables" / "wui_singlepart_summary.csv",
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
