"""Path registry and ordered task runner for fuel-reduction wildfire work."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import tasks


TASK_REGISTRY: dict[str, Callable] = {
    "validate_project_layout": tasks.validate_project_layout,
    "build_wui_singlepart_fireface_regression_data": tasks.build_wui_singlepart_fireface_regression_data,
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
        "treatment_polygons": (
            project_root / "data" / "raw" / "FACTS" / "Actv_HazFuelTrt_PL" / "Actv_HazFuelTrt_PL.shp"
        ),
        "mtbs_perimeters": project_root / "data" / "raw" / "MTBS" / "mtbs_perims_DD.shp",
        "mtbs_fod_points": (
            project_root / "data" / "raw" / "MTBS" / "mtbs_fod_pts_data" / "mtbs_FODpoints_DD.shp"
        ),
        "mtbs_bsmosaics": project_root / "data" / "raw" / "MTBS" / "MTBS_BSmosaics",
        "wui_geodatabase": (
            project_root
            / "data"
            / "raw"
            / "silvis_wui"
            / "CONUS_wui_block_1990_2020_change_v4_fgdb"
            / "CONUS_WUI_block_1990_2020_change_v4_web.gdb"
        ),
        "processed": project_root / "data" / "processed",
        "data_archive": project_root / "data" / "archive",
        "project": project_dir,
        "intermediate": project_dir / "intermediate",
        "output": output_dir,
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "report": output_dir / "report",
        "presentation": output_dir / "presentation",
        "logs": output_dir / "logs",
        "configuration_status": output_dir / "tables" / "configuration_status.csv",
        "legacy_archive": code_root / "archieve",
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
