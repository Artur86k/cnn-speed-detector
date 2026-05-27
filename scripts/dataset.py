"""
Windowing helper for the preprocessed merged.npz produced by preprocess.py.

Builds (X, y, meta) tensors of fixed-length sliding windows for CNN/TCN
training. Does NOT cross session boundaries. Only emits windows whose
samples are all `valid_label==True` (configurable threshold).

Usage as a library:
    from dataset import make_windows
    X, y, meta = make_windows(
        npz_path='samples/preprocessed/merged.npz',
        channels=('accel_veh', 'gyro'),
        window_s=2.0,
        stride_s=0.5,
        label='center',
    )

Usage as a CLI (writes a windows.npz next to the input):
    python scripts/dataset.py --window-s 2.0 --stride-s 0.5 \
        --channels accel_veh gyro --label center
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_NPZ = r'E:\Project\cnn_speed_detector\samples\preprocessed\merged.npz'

# Map of channel name -> column count (matches preprocess.py schema)
CHANNEL_DIMS = {
    'accel_phone': 3, 'gyro': 3, 'linacc': 3,
    'accel_enu':   3, 'accel_veh': 3,
    'azimuth_deg': 1, 'pitch_deg':  1, 'roll_deg': 1,
    'heading_deg': 1,
}


def _as_2d(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1, 1) if a.ndim == 1 else a


def make_windows(
    npz_path: str | Path = DEFAULT_NPZ,
    channels: tuple[str, ...] = ('accel_veh', 'gyro'),
    window_s: float = 2.0,
    stride_s: float = 0.5,
    label: str = 'center',           # 'center' | 'last' | 'mean'
    min_valid_frac: float = 1.0,     # require all samples valid by default
    target_hz: float | None = None,  # override; otherwise read from NPZ
):
    """Build sliding-window training tensors.

    Returns
    -------
    X     : (n_windows, T, C) float32
    y     : (n_windows,)      float32  — speed_ms label
    meta  : dict of (n_windows,) arrays — t_start_s, t_label_s, session_id, valid_frac
    """
    d = np.load(npz_path, allow_pickle=True)
    hz = float(d['target_hz']) if target_hz is None else float(target_hz)
    T  = int(round(window_s * hz))
    S  = max(1, int(round(stride_s * hz)))
    if T < 2:
        raise ValueError(f"window_s={window_s} too small at {hz} Hz")

    # Resolve channels into a single (N, C_total) matrix.
    for ch in channels:
        if ch not in d.files:
            raise KeyError(f"channel '{ch}' not in NPZ; have {list(d.files)}")
    cols = [_as_2d(d[ch]) for ch in channels]
    feats = np.concatenate(cols, axis=1).astype(np.float32, copy=False)   # (N, C)
    speed = d['speed_ms'].astype(np.float32)
    valid = d['valid_label'].astype(bool)
    sess  = d['session_id'].astype(np.int16)
    t_s   = d['t_s'].astype(np.float64)

    # Find runs of constant session_id so windows don't cross boundaries.
    bounds = np.flatnonzero(np.diff(sess) != 0) + 1
    starts = np.concatenate([[0], bounds])
    ends   = np.concatenate([bounds, [len(sess)]])

    X_list, y_list = [], []
    t0_list, tl_list, sid_list, vf_list = [], [], [], []

    label_idx_in_window = {'center': T // 2, 'last': T - 1}.get(label, None)
    if label not in ('center', 'last', 'mean'):
        raise ValueError(f"label must be 'center'|'last'|'mean', got {label!r}")

    for s0, s1 in zip(starts, ends):
        n_sess = s1 - s0
        if n_sess < T:
            continue
        # window start indices within this session
        wstarts = np.arange(0, n_sess - T + 1, S) + s0    # absolute indices
        # build (n_w, T) index grid
        idx = wstarts[:, None] + np.arange(T)[None, :]     # (n_w, T)

        valid_w   = valid[idx]                              # (n_w, T)
        frac      = valid_w.mean(axis=1)
        keep      = frac >= min_valid_frac
        if not keep.any():
            continue
        idx       = idx[keep]
        wstarts   = wstarts[keep]
        frac      = frac[keep]

        # gather features
        Xw = feats[idx]                                     # (n_w, T, C)
        # label
        if label == 'mean':
            yw = speed[idx].mean(axis=1)
        else:
            yw = speed[idx[:, label_idx_in_window]]

        X_list.append(Xw)
        y_list.append(yw.astype(np.float32))
        t0_list.append(t_s[wstarts].astype(np.float32))
        if label == 'mean':
            tl_list.append(t_s[idx].mean(axis=1).astype(np.float32))
        else:
            tl_list.append(t_s[idx[:, label_idx_in_window]].astype(np.float32))
        sid_list.append(sess[wstarts].astype(np.int16))
        vf_list.append(frac.astype(np.float32))

    if not X_list:
        raise RuntimeError("No valid windows produced. Lower min_valid_frac or window length.")

    X    = np.concatenate(X_list, axis=0)
    y    = np.concatenate(y_list, axis=0)
    meta = dict(
        t_start_s = np.concatenate(t0_list),
        t_label_s = np.concatenate(tl_list),
        session_id= np.concatenate(sid_list),
        valid_frac= np.concatenate(vf_list),
        channels  = list(channels),
        window_s  = float(window_s),
        stride_s  = float(stride_s),
        target_hz = hz,
        label     = label,
    )
    return X, y, meta


def session_split(meta: dict, val_sessions: tuple[int, ...] = (1,)) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, val_mask) selecting windows by session id."""
    sid = meta['session_id']
    val = np.isin(sid, val_sessions)
    return ~val, val


def make_windows_from_dir(
    npz_dir: str | Path,
    channels: tuple[str, ...] = ('accel_veh', 'gyro'),
    window_s: float = 2.0,
    stride_s: float = 0.5,
    label: str = 'center',
    min_valid_frac: float = 0.95,
    min_session_valid_frac: float = 0.10,  # skip sessions with <10% valid samples
    blocklist: tuple[str, ...] = (),       # session NAMES (without .npz) to exclude
):
    """Walk a directory of per-session NPZ files, build windows across them.

    A session_id is assigned in the order files are discovered (sorted by name).
    Sessions whose own valid_label fraction is below min_session_valid_frac
    are skipped entirely (poor-quality drives waste training time).

    Returns (X, y, meta) with the same shape as make_windows().
    meta also includes 'session_names' (kept-only) so you can map session_id
    back to the original recording.
    """
    npz_dir = Path(npz_dir)
    # Skip non-session NPZs that may sit alongside the per-session files.
    SKIP_STEMS = {'merged', 'windows', 'train_history'}
    files = sorted(
        p for p in npz_dir.glob('*.npz')
        if p.stem not in blocklist and p.stem not in SKIP_STEMS
    )

    X_list, y_list = [], []
    t0_list, tl_list, sid_list, vf_list = [], [], [], []
    kept_names: list[str] = []
    skipped: list[tuple[str, str]] = []

    label_idx_in_window = None  # resolved after we know T
    T = S = None
    hz = None

    sid_next = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        if hz is None:
            hz = float(d['target_hz'])
            T  = int(round(window_s * hz))
            S  = max(1, int(round(stride_s * hz)))
            label_idx_in_window = {'center': T // 2, 'last': T - 1}.get(label, None)
            if label not in ('center', 'last', 'mean'):
                raise ValueError(f"label must be 'center'|'last'|'mean', got {label!r}")

        valid = d['valid_label'].astype(bool)
        if valid.size == 0 or valid.mean() < min_session_valid_frac:
            skipped.append((f.stem, f"valid={100*valid.mean():.1f}%"))
            continue

        for ch in channels:
            if ch not in d.files:
                raise KeyError(f"{f.name}: channel {ch!r} missing")
        feats = np.concatenate([_as_2d(d[ch]) for ch in channels], axis=1).astype(np.float32, copy=False)
        speed = d['speed_ms'].astype(np.float32)
        t_s   = d['t_s'].astype(np.float64)
        N     = feats.shape[0]

        if N < T:
            skipped.append((f.stem, f"shorter than window ({N} < {T})"))
            continue

        wstarts = np.arange(0, N - T + 1, S)
        idx     = wstarts[:, None] + np.arange(T)[None, :]
        valid_w = valid[idx]
        frac    = valid_w.mean(axis=1)
        keep    = frac >= min_valid_frac
        if not keep.any():
            skipped.append((f.stem, "no fully-valid windows"))
            continue
        idx     = idx[keep]
        wstarts = wstarts[keep]
        frac    = frac[keep]

        Xw = feats[idx]
        yw = speed[idx].mean(axis=1) if label == 'mean' else speed[idx[:, label_idx_in_window]]

        X_list.append(Xw)
        y_list.append(yw.astype(np.float32))
        t0_list.append(t_s[wstarts].astype(np.float32))
        tl_list.append(
            t_s[idx].mean(axis=1).astype(np.float32) if label == 'mean'
            else t_s[idx[:, label_idx_in_window]].astype(np.float32)
        )
        sid_list.append(np.full(Xw.shape[0], sid_next, dtype=np.int16))
        vf_list.append(frac.astype(np.float32))
        kept_names.append(f.stem)
        print(f"  {f.stem:38s} -> {Xw.shape[0]:6d} windows  (valid {100*valid.mean():.1f}%)")
        sid_next += 1

    if skipped:
        print(f"\n  Skipped {len(skipped)} sessions:")
        for n, why in skipped:
            print(f"    {n:38s}  {why}")

    if not X_list:
        raise RuntimeError("No usable sessions found.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    meta = dict(
        t_start_s     = np.concatenate(t0_list),
        t_label_s     = np.concatenate(tl_list),
        session_id    = np.concatenate(sid_list),
        valid_frac    = np.concatenate(vf_list),
        session_names = kept_names,
        channels      = list(channels),
        window_s      = float(window_s),
        stride_s      = float(stride_s),
        target_hz     = hz,
        label         = label,
    )
    return X, y, meta


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--npz',  default=None,
                     help='single merged.npz (single-session-corpus mode)')
    src.add_argument('--dir',  default=None,
                     help='directory of per-session NPZ files (multi-session mode)')
    ap.add_argument('--out',      default=None, help='output .npz')
    ap.add_argument('--channels', nargs='+', default=['accel_veh', 'gyro'],
                    choices=list(CHANNEL_DIMS.keys()))
    ap.add_argument('--window-s', type=float, default=2.0)
    ap.add_argument('--stride-s', type=float, default=0.5)
    ap.add_argument('--label',    choices=['center', 'last', 'mean'], default='center')
    ap.add_argument('--min-valid-frac',         type=float, default=1.0)
    ap.add_argument('--min-session-valid-frac', type=float, default=0.10,
                    help='(--dir mode) drop sessions whose own valid_label < this')
    ap.add_argument('--blocklist', nargs='+', default=[],
                    help='(--dir mode) session stems to exclude (e.g. navlog_20260515_013747)')
    args = ap.parse_args()

    if args.dir:
        X, y, meta = make_windows_from_dir(
            npz_dir                = args.dir,
            channels               = tuple(args.channels),
            window_s               = args.window_s,
            stride_s               = args.stride_s,
            label                  = args.label,
            min_valid_frac         = args.min_valid_frac,
            min_session_valid_frac = args.min_session_valid_frac,
            blocklist              = tuple(args.blocklist),
        )
        default_out = Path(args.dir) / 'windows.npz'
    else:
        npz_path = args.npz or DEFAULT_NPZ
        X, y, meta = make_windows(
            npz_path        = npz_path,
            channels        = tuple(args.channels),
            window_s        = args.window_s,
            stride_s        = args.stride_s,
            label           = args.label,
            min_valid_frac  = args.min_valid_frac,
        )
        default_out = Path(npz_path).with_name('windows.npz')

    n_w, T, C = X.shape
    print(f"Windows : {n_w}  shape=(T={T}, C={C})  channels={meta['channels']}")
    print(f"Label   : {args.label}  y min={y.min():.2f}  max={y.max():.2f}  mean={y.mean():.2f}  median={float(np.median(y)):.2f}")
    sid = meta['session_id']
    for s in np.unique(sid):
        print(f"  session {s}: {int((sid==s).sum())} windows")

    out = Path(args.out) if args.out else default_out
    np.savez_compressed(out, X=X, y=y, **{k: v for k, v in meta.items()
                                          if isinstance(v, np.ndarray)},
                        channels=np.array(meta['channels']),
                        window_s=np.float32(meta['window_s']),
                        stride_s=np.float32(meta['stride_s']),
                        target_hz=np.float32(meta['target_hz']),
                        label=np.array(meta['label']))
    print(f"Wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
