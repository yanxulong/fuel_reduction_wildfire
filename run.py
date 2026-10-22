"""Entry point for the local and MSI project workflow."""

from __future__ import annotations

import argparse
import sys

sys.dont_write_bytecode = True

import initialization


CONFIG = {
    "enabled_tasks": ["build_wui_singlepart_dataset"],
    "overwrite_existing": False,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_tasks:
        for task_name in initialization.TASK_REGISTRY:
            print(task_name)
        return 0

    config = dict(CONFIG)
    config["enabled_tasks"] = args.task or list(CONFIG["enabled_tasks"])
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
