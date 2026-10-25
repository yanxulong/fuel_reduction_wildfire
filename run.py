"""Entry point for the local and MSI project workflow."""

from __future__ import annotations

import argparse
import sys

sys.dont_write_bytecode = True

import initialization


CONFIG = {
    "enabled_tasks": [
        "build_wui_singlepart_dataset",
        "build_fire_wui_geometry_characteristics",
    ],
    "overwrite_existing": False,
    "fire_wui_distance_miles": 3.0,
    "treatment_rolling_years": list(range(1, 11)),
    "treatment_lag_bands": [
        ("lag_1_2y", 1, 2),
        ("lag_2_4y", 2, 4),
    ],
    "treatment_angular_thresholds": [pct / 100 for pct in range(5, 101, 5)],
    "treatment_wui_buffer_miles": None,
    "fire_perimeter_threat_buffer_miles": 1.5,
    "treatment_thickness_step_degrees": 0.25,
    "treatment_pair_batch_size": 256,
    "treatment_array_worker_id": None,
    "treatment_array_worker_count": None,
    "treatment_array_run_id": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Set FUEL_REDUCTION_PROJECT_ROOT to override the project/data/output root. "
        "Default: parent directory of this code checkout.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=sorted(initialization.TASK_REGISTRY),
        help="Task to run; repeat flag to select multiple tasks.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="List registered tasks and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Show paths and task order without running tasks.")
    parser.add_argument(
        "--distance-miles",
        type=float,
        default=None,
        help="Fire-centroid to WUI single-part edge search distance (miles; default: 3).",
    )
    parser.add_argument("--array-worker-id", type=int, default=None, help="Array worker index for treatment shard mode.")
    parser.add_argument("--array-worker-count", type=int, default=None, help="Total treatment array worker count.")
    parser.add_argument("--array-run-id", default=None, help="Unique run ID shared by treatment array workers and finalizer.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Replace complete outputs during finalization.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_tasks:
        for task_name in initialization.TASK_REGISTRY:
            print(task_name)
        return 0

    config = dict(CONFIG)
    config["enabled_tasks"] = args.task or list(CONFIG["enabled_tasks"])
    config["treatment_array_worker_id"] = args.array_worker_id
    config["treatment_array_worker_count"] = args.array_worker_count
    config["treatment_array_run_id"] = args.array_run_id
    if args.overwrite_existing:
        config["overwrite_existing"] = True
    if args.distance_miles is not None:
        config["fire_wui_distance_miles"] = args.distance_miles
    array_options_given = any((args.array_worker_id is not None, args.array_worker_count is not None, args.array_run_id))
    if array_options_given and not args.array_run_id:
        raise SystemExit("--array-run-id is required with treatment array options.")
    if args.array_worker_id is not None:
        if args.array_worker_count is None or args.array_worker_count < 1:
            raise SystemExit("--array-worker-count must be positive when --array-worker-id is supplied.")
        if config["enabled_tasks"] != ["build_fire_wui_treatment_definitions"]:
            raise SystemExit("Array worker mode requires only --task build_fire_wui_treatment_definitions.")
    elif array_options_given and config["enabled_tasks"] != ["finalize_fire_wui_treatment_definitions"]:
        raise SystemExit("Array metadata without a worker ID is reserved for the treatment finalizer task.")
    if "finalize_fire_wui_treatment_definitions" in config["enabled_tasks"] and args.array_worker_count is None:
        raise SystemExit("Finalizer requires --array-worker-count and --array-run-id.")
    if config.get("overwrite_existing", False):
        print("[policy] overwrite enabled; existing task outputs may be rebuilt.")
    else:
        print("[policy] overwrite disabled; completed task outputs will be skipped.")
    project = initialization.LocalProjectFlow()
    initialization.initialize_paths(project, config)

    if args.dry_run:
        print("Resolved paths:")
        for path_name, path in project.paths.items():
            print(f"  {path_name}: {path}")
        print("Task order:")
        for index, task_name in enumerate(config["enabled_tasks"], start=1):
            print(f"  {index}. {task_name}")
        return 0

    initialization.build_task_tree(project, config)
    project.execute()
    print("[complete] Workflow finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
