"""Small, observable tasks for the fuel-reduction wildfire workflow."""

from __future__ import annotations

from datetime import datetime, timezone

import functions


def validate_project_layout(project) -> dict[str, object]:
    """Check workspace locations and write a compact status report."""
    print("[stage] Checking code checkout and project folders.")
    required = ("code", "project_root")
    for key in required:
        if not project.paths[key].is_dir():
            raise FileNotFoundError(f"Required project path is missing: {project.paths[key]}")

    for key in ("output", "tables", "logs"):
        functions.ensure_dir(project.paths[key])

    checks = [
        ("code_checkout", "code", True),
        ("project_root", "project_root", True),
        ("raw_data", "raw", False),
        ("processed_data", "processed", False),
        ("archived_data", "data_archive", False),
        ("legacy_code_archive", "legacy_archive", False),
        ("project_output", "output", True),
    ]
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    missing_optional = []
    for check_name, path_key, is_required in checks:
        path = project.paths[path_key]
        exists = path.is_dir()
        status = "present" if exists else ("missing_required" if is_required else "missing_optional")
        detail = "Directory is available." if exists else "Directory is not present yet."
        rows.append(
            {
                "checked_at_utc": timestamp,
                "check": check_name,
                "status": status,
                "path": str(path),
                "detail": detail,
            }
        )
        print(f"[check] {check_name}: {status} | {path}")
        if not exists and not is_required:
            missing_optional.append(path_key)

    report_path = functions.write_csv(
        rows,
        project.paths["configuration_status"],
        fieldnames=("checked_at_utc", "check", "status", "path", "detail"),
    )
    status = "PASS_WITH_WARNINGS" if missing_optional else "PASS"
    print(f"[result] {status}; {len(rows)} checks; {len(missing_optional)} optional paths missing.")
    print(f"[output] {report_path}")
    return {"status": status, "report": report_path, "missing_optional": missing_optional}
