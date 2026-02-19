from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

def _load_mat(path: Path) -> dict:
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    except TypeError:
        return loadmat(path, squeeze_me=True, struct_as_record=False)

def convert_mat_dir_to_csv_two_channels(mat_dir: str | Path, out_dir: str | Path) -> None:
    mat_dir = Path(mat_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mats = sorted(mat_dir.glob("*.mat"))
    if not mats:
        raise FileNotFoundError(f"No .mat files found in {mat_dir}")

    for mp in mats:
        d = _load_mat(mp)

        # Your known keys
        t0 = np.asarray(d["UntitledPXI1Slot2_ai0_Time"]).ravel()
        a0 = np.asarray(d["UntitledPXI1Slot2_ai0"]).ravel()
        t1 = np.asarray(d["UntitledPXI1Slot2_ai1_Time"]).ravel()
        a1 = np.asarray(d["UntitledPXI1Slot2_ai1"]).ravel()

        # Use ai0 time as the master time base
        if t0.size != a0.size or t1.size != a1.size:
            raise ValueError(f"{mp.name}: time/signal length mismatch")

        # Align ai1 to ai0 time if needed
        if np.allclose(t0, t1):
            a1_aligned = a1
        else:
            a1_aligned = np.interp(t0, t1, a1)

        df = pd.DataFrame({
            "Time": t0,
            "ai0": a0,
            "ai1": a1_aligned
        })

        df.to_csv(out_dir / f"{mp.stem}.csv", index=False)

    print(f"Converted {len(mats)} files -> {out_dir}")

# Example:
convert_mat_dir_to_csv_two_channels("data/MATLAB", "data/CSV")
