import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # folder containing this script

SOURCE_DIR = (BASE_DIR / "boiling_plots").resolve()
DEST_DIR   = (BASE_DIR / "rep_samples/UMAP-HDBSCAN").resolve()

print("Source dir:", SOURCE_DIR)
print("Dest dir  :", DEST_DIR)

if not DEST_DIR.exists():
    raise FileNotFoundError(f"Destination directory not found: {DEST_DIR}")
if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

replaced = kept = errors = 0

for subdir in DEST_DIR.iterdir():
    if not subdir.is_dir():
        continue

    for dest_png in subdir.glob("*.png"):
        # 1) primary: flat boiling_plots/<filename>
        source_png = SOURCE_DIR / dest_png.name

        # 2) fallback: boiling_plots/<cluster>/<filename> (in case you later add subfolders)
        if not source_png.is_file():
            alt = SOURCE_DIR / subdir.name / dest_png.name
            if alt.is_file():
                source_png = alt

        if source_png.is_file():
            try:
                shutil.copy2(source_png, dest_png)  # overwrite
                replaced += 1
                print(f"Replaced: {dest_png}  <=  {source_png}")
            except Exception as e:
                errors += 1
                print(f"ERROR replacing {dest_png} with {source_png}: {e}")
        else:
            kept += 1
            print(f"Kept (no match): {dest_png}")

print(f"\nDone. Replaced={replaced}, Kept={kept}, Errors={errors}")