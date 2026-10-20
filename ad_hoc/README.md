# Ad hoc research tasks

Use this folder for one-off data intake and maintenance tasks that do not belong in the repeatable analysis pipeline. Keep task scripts reviewable and rerunnable. Store downloaded data under the project `data/` tree, never in Git.

## SILVIS WUI Change 1990–2020, Version 4

`download_silvis_wui.sh` downloads the conterminous-U.S. File Geodatabase archive linked from the [official SILVIS WUI Change page](https://silvis.forest.wisc.edu/data/wui-change/). Product covers WUI for 1990, 2000, 2010, and 2020. It saves the archive and a verification record under the project’s `data/raw/silvis_wui/` tree. The download resumes from a partial `.part` file and checks ZIP integrity before promoting it to the final filename.

Run from the MSI checkout after syncing code:

```bash
cd /users/8/yxlong/projects/fuel_reduction_wildfire/code
bash ad_hoc/download_silvis_wui.sh
```

Check available project storage before starting. Do not commit the archive or generated metadata.
