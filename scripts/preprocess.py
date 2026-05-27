"""
Preprocess Android NavRecorder logs for speed-CNN training.

Quality criteria mirror E:/Project/offline_navigation_system/matlab/ekf_car_nav.m:
  - Stationary detection: movvar(|a|, 2 s) < 0.02 m^2/s^4
  - Gravity: mean accel over initial stationary segment; |g| within 1.0 of 9.81
  - Frame transform: phone -> ENU via gamerot azimuth/pitch/roll (mag-free)
  - Spoof tiers 1-3 (hAcc ramp, position jump, IMU/GPS heading divergence)
  - GPS gap mask: > 5 s since last good fix
  - Label gate (strict): hAcc < 15 m AND speed_acc < 0.5 m/s AND spoof_flag == 0

Outputs NPZ files at 200 Hz under samples/preprocessed/:
  - <session>.npz   : one file per session
  - merged.npz      : concatenation of all sessions with session_id channel

Usage:
  python scripts/preprocess.py            # processes all samples/navlog_*/
  python scripts/preprocess.py --no-merge # skip merged.npz
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ----- constants ported from ekf_car_nav.m -------------------------------------
TARGET_HZ        = 200.0
DT               = 1.0 / TARGET_HZ
STILL_VAR_THR    = 0.02      # m^2/s^4
STILL_WIN_SEC    = 2.0
GRAVITY_NOMINAL  = 9.81
GRAVITY_TOL      = 1.0

# spoof detector thresholds
SPOOF_HACC_RAMP_THR = 2.0    # m/s, d(hAcc)/dt per fix
SPOOF_HACC_RAMP_N   = 5
SPOOF_HACC_BAD      = 15.0   # ramp only counts when hAcc already > 15 m
SPOOF_POS_JUMP_MS   = 50.0   # impossible inter-fix speed
SPOOF_HEADING_DEG   = 20.0   # IMU-vs-GPS heading divergence
SPOOF_RECOVERY_N    = 30
SPOOF_MAX_SPEED_MS  = 25.0   # m/s, recovery reachability envelope

# label gate (strict, per user)
LABEL_HACC_MAX     = 15.0
LABEL_SPDACC_MAX   = 0.5
GPS_GAP_THR_S      = 5.0

EARTH_R            = 6378137.0

# ----- data classes ------------------------------------------------------------
@dataclass
class Session:
    name: str
    t_s: np.ndarray            # (N,) common 200 Hz timeline, seconds from session t0
    # phone body frame
    accel_phone: np.ndarray    # (N, 3) raw accel (gravity included), m/s^2
    gyro:        np.ndarray    # (N, 3) rad/s
    linacc:      np.ndarray    # (N, 3) Android linear accel (gravity removed), m/s^2
    # ENU world frame (gravity removed, rotated via gamerot)
    accel_enu:   np.ndarray    # (N, 3) [east, north, up]
    # vehicle body frame (forward = heading, lateral = right, vertical = up)
    accel_veh:   np.ndarray    # (N, 3) [fwd, lat, vert]
    # orientation
    azimuth_deg: np.ndarray    # (N,) gamerot azimuth, 0=N CW
    pitch_deg:   np.ndarray    # (N,)
    roll_deg:    np.ndarray    # (N,)
    heading_deg: np.ndarray    # (N,) GPS course where moving, gamerot az fallback
    # labels and masks
    speed_ms:    np.ndarray    # (N,) interpolated GPS speed
    speed_acc_ms:np.ndarray    # (N,) interpolated GPS speed accuracy
    hacc_m:      np.ndarray    # (N,) interpolated GPS horizontal accuracy
    valid_label: np.ndarray    # (N,) bool: strict gate satisfied AND not in gap
    in_gap:      np.ndarray    # (N,) bool: > 5 s since last good fix
    stationary:  np.ndarray    # (N,) bool: |a|-variance below threshold
    spoof_flag:  np.ndarray    # (N,) int8 nearest-neighbor of GPS spoof state
    # gravity used (for diagnostics)
    g_phone:     np.ndarray    # (3,) phone-frame gravity vector


# ----- helpers -----------------------------------------------------------------
def _read_csv(p: Path, dtype_map: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(p)
    if dtype_map:
        for c, t in dtype_map.items():
            if c in df.columns:
                df[c] = df[c].astype(t)
    return df


def _interp(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear interp with endpoint hold (matches MATLAB interp1 + clamping)."""
    t_dst = np.clip(t_dst, t_src[0], t_src[-1])
    return np.interp(t_dst, t_src, y_src)


def _movvar(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving variance, MATLAB movvar default (edges shorten window)."""
    if win < 2:
        return np.zeros_like(x)
    s  = pd.Series(x)
    half = win // 2
    return s.rolling(window=win, min_periods=max(2, half), center=True).var().bfill().ffill().to_numpy()


def _wrap_angle_deg(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


# ----- per-session pipeline ----------------------------------------------------
def detect_stationary(t_a_s: np.ndarray, accel: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Returns (is_still per IMU sample, walk_start_idx, last_still_idx)."""
    dt_imu = float(np.median(np.diff(t_a_s)))
    win    = max(2, int(round(STILL_WIN_SEC / dt_imu)))
    a_mag  = np.linalg.norm(accel, axis=1)
    var    = _movvar(a_mag, win)
    is_still = var < STILL_VAR_THR
    first_still = int(np.argmax(is_still)) if is_still.any() else 0
    rest        = ~is_still[first_still:]
    walk_start  = int(np.argmax(rest)) + first_still if rest.any() else first_still
    last_still  = int(len(is_still) - 1 - np.argmax(is_still[::-1])) if is_still.any() else len(is_still) - 1
    return is_still, walk_start, last_still


def estimate_gravity(accel: np.ndarray, walk_start: int) -> np.ndarray:
    """Mean of initial stationary samples = phone-frame gravity vector."""
    if walk_start < 10:
        # no clear stationary start; use the lowest-variance 2 s window
        a_mag = np.linalg.norm(accel, axis=1)
        var   = _movvar(a_mag, max(2, int(round(STILL_WIN_SEC * 100))))
        c     = int(np.argmin(var))
        lo, hi = max(0, c - 100), min(len(accel), c + 100)
        return accel[lo:hi].mean(axis=0)
    return accel[:walk_start].mean(axis=0)


def rotate_phone_to_enu(a_phone: np.ndarray, az_deg: np.ndarray,
                        pit_deg: np.ndarray, rol_deg: np.ndarray) -> np.ndarray:
    """Rotate phone-body vectors to ENU using gamerot azimuth/pitch/roll.

    Same R = Rz(yaw_enu) * Ry(pitch) * Rx(roll) used in ekf_car_nav.m, extended
    with the Up row so we keep the vertical channel for the NN.
    """
    yaw = np.deg2rad(az_deg)
    pit = np.deg2rad(pit_deg)
    rol = np.deg2rad(rol_deg)
    yaw_enu = np.pi / 2 - yaw

    cy, sy = np.cos(yaw_enu), np.sin(yaw_enu)
    cp, sp = np.cos(pit),     np.sin(pit)
    cr, sr = np.cos(rol),     np.sin(rol)

    ax, ay, az = a_phone[:, 0], a_phone[:, 1], a_phone[:, 2]
    a_e = cy*cp*ax + (cy*sp*sr - sy*cr)*ay + (cy*sp*cr + sy*sr)*az
    a_n = sy*cp*ax + (sy*sp*sr + cy*cr)*ay + (sy*sp*cr - cy*sr)*az
    a_u = -sp*ax  +  cp*sr*ay              +  cp*cr*az
    return np.stack([a_e, a_n, a_u], axis=1)


def rotate_enu_to_vehicle(a_enu: np.ndarray, heading_deg: np.ndarray) -> np.ndarray:
    """ENU -> [forward, lateral_right, vertical] where forward aligns with heading.

    Compass course (0 = N, CW). ENU axes are East/North/Up.
        forward_e = sin(course),   forward_n = cos(course)
        right_e   = cos(course),   right_n   = -sin(course)
    """
    h = np.deg2rad(heading_deg)
    s, c = np.sin(h), np.cos(h)
    a_fwd = a_enu[:, 0] *  s + a_enu[:, 1] *  c
    a_lat = a_enu[:, 0] *  c + a_enu[:, 1] * -s
    a_vrt = a_enu[:, 2]
    return np.stack([a_fwd, a_lat, a_vrt], axis=1)


def latlon_to_xy(lat: np.ndarray, lon: np.ndarray,
                 lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular ENU around (lat0, lon0)."""
    lat0r = np.deg2rad(lat0)
    x = np.deg2rad(lon - lon0) * EARTH_R * np.cos(lat0r)
    y = np.deg2rad(lat - lat0) * EARTH_R
    return x, y


def detect_spoof(t_p_s: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                 hAcc: np.ndarray, spd: np.ndarray, crs: np.ndarray,
                 imu_az_at_gps: np.ndarray) -> np.ndarray:
    """Run MATLAB Tiers 1-3 spoof detector. Returns spoof_flag per GPS fix.
    0 = ok, 1 = suspect (hAcc ramp), 2 = confirmed spoof, 3 = hard reject.
    """
    n = len(t_p_s)
    flag = np.zeros(n, dtype=np.int8)
    state = 0
    hacc_ramp_n = 0
    head_div_n  = 0
    recovery_n  = 0
    last_good_i = 0

    for i in range(1, n):
        dt = t_p_s[i] - t_p_s[i-1]
        if dt <= 0:
            flag[i] = state
            continue

        # ---- Tier 1: hAcc ramp ----
        dhacc_dt = (hAcc[i] - hAcc[i-1]) / dt
        if dhacc_dt > SPOOF_HACC_RAMP_THR and hAcc[i] > SPOOF_HACC_BAD:
            hacc_ramp_n += 1
        else:
            hacc_ramp_n = max(0, hacc_ramp_n - 1)
        if hacc_ramp_n >= SPOOF_HACC_RAMP_N and state < 1:
            state = 1
        elif hacc_ramp_n == 0 and state == 1:
            state = 0

        # ---- Tier 2: position jump ----
        dlat = np.deg2rad(lat[i] - lat[i-1])
        dlon = np.deg2rad(lon[i] - lon[i-1])
        latm = np.deg2rad((lat[i] + lat[i-1]) / 2)
        d_pos = EARTH_R * np.sqrt(dlat**2 + (np.cos(latm) * dlon)**2)
        pos_sp = d_pos / dt
        hAcc_gate = 3 * max(hAcc[i], hAcc[i-1])
        if pos_sp > SPOOF_POS_JUMP_MS and d_pos > hAcc_gate:
            state = 2
            flag[i] = 3
            continue

        # ---- Tier 3: IMU heading divergence (only escalates 1 -> 2) ----
        if spd[i] > 1.0 and spd[i-1] > 1.0:
            d_bear = ((crs[i] - crs[i-1] + 180.0) % 360.0) - 180.0
            d_imu  = ((imu_az_at_gps[i] - imu_az_at_gps[i-1] + 180.0) % 360.0) - 180.0
            hd     = abs(d_bear - d_imu)
            if hd > SPOOF_HEADING_DEG and state == 1:
                head_div_n += 1
            else:
                head_div_n = max(0, head_div_n - 1)
            if head_div_n >= SPOOF_HACC_RAMP_N and state == 1:
                state = 2
        else:
            head_div_n = max(0, head_div_n - 1)

        # ---- Recovery ----
        if state >= 2:
            is_clean = hAcc[i] < 25
            t_since   = t_p_s[i] - t_p_s[last_good_i]
            dlat_rc   = np.deg2rad(lat[i] - lat[last_good_i])
            dlon_rc   = np.deg2rad(lon[i] - lon[last_good_i])
            latm_rc   = np.deg2rad((lat[i] + lat[last_good_i]) / 2)
            d_from_good = EARTH_R * np.sqrt(dlat_rc**2 + (np.cos(latm_rc) * dlon_rc)**2)
            max_reach   = t_since * SPOOF_MAX_SPEED_MS + hAcc[i] + hAcc[last_good_i]
            is_reach    = d_from_good <= max_reach
            if is_clean and is_reach:
                recovery_n += 1
                if recovery_n >= SPOOF_RECOVERY_N:
                    state = 0
                    hacc_ramp_n = 0
                    head_div_n  = 0
                    recovery_n  = 0
            else:
                recovery_n = 0

        flag[i] = state
        if flag[i] == 0:
            last_good_i = i
    return flag


def preprocess_session(folder: Path) -> Session:
    print(f"\n=== {folder.name} ===", flush=True)

    accel    = _read_csv(folder / 'accel.csv')
    gyro     = _read_csv(folder / 'gyro.csv')
    linacc   = _read_csv(folder / 'linacc.csv')
    gamerot  = _read_csv(folder / 'gamerot.csv')
    gnss     = _read_csv(folder / 'gnss.csv')

    # ---- Build IMU timeline. All sensors share SystemClock.elapsedRealtimeNanos. ----
    t0_ns = int(accel['elapsed_ns'].iloc[0])
    def sec(df): return (df['elapsed_ns'].astype('int64').to_numpy() - t0_ns) / 1e9

    t_a = sec(accel)
    t_g = sec(gyro)
    t_l = sec(linacc)
    t_o = sec(gamerot)
    t_p = sec(gnss)

    a_phone_raw = accel[['ax_ms2', 'ay_ms2', 'az_ms2']].to_numpy()
    g_phone_raw = gyro [['gx_rads','gy_rads','gz_rads']].to_numpy()
    l_phone_raw = linacc[['ax_ms2','ay_ms2','az_ms2']].to_numpy()
    az_raw = gamerot['azimuth_deg'].to_numpy()
    pi_raw = gamerot['pitch_deg'].to_numpy()
    ro_raw = gamerot['roll_deg'].to_numpy()

    # ---- Stationary / gravity (on raw IMU rate) ----
    is_still_imu, walk_start, last_still = detect_stationary(t_a, a_phone_raw)
    g_phone = estimate_gravity(a_phone_raw, walk_start)
    g_norm  = float(np.linalg.norm(g_phone))
    print(f"  gravity phone={g_phone.round(3).tolist()}  |g|={g_norm:.3f}")
    if abs(g_norm - GRAVITY_NOMINAL) > GRAVITY_TOL:
        print(f"  WARNING: |g| deviates from 9.81 by {abs(g_norm - GRAVITY_NOMINAL):.2f}")
    a_phone_lin_raw = a_phone_raw - g_phone

    # ---- Spoof detection on raw GNSS arrays ----
    # interp gamerot azimuth to GPS fix timestamps
    t_o_u, idx_u = np.unique(t_o, return_index=True)
    az_at_gps = _interp(t_o_u, az_raw[idx_u], t_p)
    hAcc_arr = gnss['hacc_m'].to_numpy()
    spd_arr  = gnss['speed_ms'].to_numpy()
    crs_arr  = gnss['bearing_deg'].to_numpy()
    sacc_arr = gnss['speed_acc_ms'].to_numpy()
    lat_arr  = gnss['latitude'].to_numpy()
    lon_arr  = gnss['longitude'].to_numpy()
    spoof = detect_spoof(t_p, lat_arr, lon_arr, hAcc_arr, spd_arr, crs_arr, az_at_gps)
    print(f"  spoof: ok={int((spoof==0).sum())}  suspect={int((spoof==1).sum())}  "
          f"spoofed={int((spoof==2).sum())}  hard-rej={int((spoof==3).sum())}")

    # ---- Build common 200 Hz timeline ----
    t_end = min(t_a[-1], t_g[-1], t_l[-1], t_o[-1])
    t_s = np.arange(0.0, t_end, DT)
    N   = len(t_s)
    print(f"  duration={t_end:.1f}s  samples@200Hz={N}")

    # ---- Resample IMU & orientation ----
    def vec_interp(t_src, arr, t_dst):
        return np.stack([_interp(t_src, arr[:, i], t_dst) for i in range(arr.shape[1])], axis=1)

    accel_phone = vec_interp(t_a, a_phone_raw,     t_s)
    gyro_r      = vec_interp(t_g, g_phone_raw,     t_s)
    linacc_r    = vec_interp(t_l, l_phone_raw,     t_s)
    accel_lin_r = vec_interp(t_a, a_phone_lin_raw, t_s)   # gravity-removed by mean

    az_r = _interp(t_o, az_raw, t_s)
    pi_r = _interp(t_o, pi_raw, t_s)
    ro_r = _interp(t_o, ro_raw, t_s)

    # stationary on 200 Hz grid (nearest-neighbor of IMU mask)
    is_still_r = _interp(t_a, is_still_imu.astype(float), t_s) > 0.5

    # ---- Rotate to ENU and vehicle frames ----
    accel_enu = rotate_phone_to_enu(accel_lin_r, az_r, pi_r, ro_r)

    # ---- GPS labels (interp onto 200 Hz, mask) ----
    if len(t_p) < 2:
        raise ValueError(f"{folder.name}: < 2 GNSS fixes, cannot label")

    # Use only non-spoofed fixes for label interpolation
    good_for_label = (
        (hAcc_arr < LABEL_HACC_MAX) &
        (sacc_arr < LABEL_SPDACC_MAX) &
        (spoof    == 0)
    )
    # Always use clean fixes also for heading where moving
    moving = good_for_label & (spd_arr > 1.0)

    speed_ms = np.zeros(N)
    if good_for_label.sum() >= 2:
        speed_ms = _interp(t_p[good_for_label], spd_arr[good_for_label], t_s)
    elif good_for_label.sum() == 1:
        speed_ms[:] = spd_arr[good_for_label][0]

    speed_acc_ms = _interp(t_p, sacc_arr, t_s)
    hacc_m       = _interp(t_p, hAcc_arr, t_s)

    # ---- Heading: GPS course where moving, gamerot azimuth fallback ----
    if moving.sum() >= 2:
        # 'previous' interpolation for course (matches MATLAB)
        t_mov   = t_p[moving]
        crs_mov = crs_arr[moving]
        idx     = np.searchsorted(t_mov, t_s, side='right') - 1
        idx     = np.clip(idx, 0, len(t_mov) - 1)
        heading_gps = crs_mov[idx]
        # before first moving fix, fall back to gamerot
        before_first = t_s < t_mov[0]
        heading_deg  = np.where(before_first, az_r, heading_gps)
    else:
        heading_deg = az_r.copy()

    accel_veh = rotate_enu_to_vehicle(accel_enu, heading_deg)

    # ---- GPS gap mask ----
    # last_good_time(t) = max t_p[good_for_label] with t_p <= t
    if good_for_label.any():
        t_good_fix = t_p[good_for_label]
        idx        = np.searchsorted(t_good_fix, t_s, side='right') - 1
        last_good  = np.where(idx >= 0, t_good_fix[np.clip(idx, 0, None)], -1e9)
        in_gap     = (t_s - last_good) > GPS_GAP_THR_S
    else:
        in_gap = np.ones(N, dtype=bool)

    # ---- Validity for the label: strict gate AND not in gap ----
    valid_label = (~in_gap) & (hacc_m < LABEL_HACC_MAX) & (speed_acc_ms < LABEL_SPDACC_MAX)

    # ---- Spoof flag on 200 Hz grid: previous-fix hold ----
    # (each grid sample inherits the flag of the most recent GPS fix at or before it)
    spoof_r_idx = np.searchsorted(t_p, t_s, side='right') - 1
    spoof_r_idx = np.clip(spoof_r_idx, 0, len(t_p) - 1)
    spoof_grid  = spoof[spoof_r_idx]

    label_pct = 100.0 * valid_label.mean()
    print(f"  valid_label coverage: {label_pct:.1f}%  ({int(valid_label.sum())}/{N})")

    return Session(
        name=folder.name,
        t_s=t_s.astype(np.float64),
        accel_phone=accel_phone.astype(np.float32),
        gyro=gyro_r.astype(np.float32),
        linacc=linacc_r.astype(np.float32),
        accel_enu=accel_enu.astype(np.float32),
        accel_veh=accel_veh.astype(np.float32),
        azimuth_deg=az_r.astype(np.float32),
        pitch_deg=pi_r.astype(np.float32),
        roll_deg=ro_r.astype(np.float32),
        heading_deg=heading_deg.astype(np.float32),
        speed_ms=speed_ms.astype(np.float32),
        speed_acc_ms=speed_acc_ms.astype(np.float32),
        hacc_m=hacc_m.astype(np.float32),
        valid_label=valid_label,
        in_gap=in_gap,
        stationary=is_still_r,
        spoof_flag=spoof_grid.astype(np.int8),
        g_phone=g_phone.astype(np.float32),
    )


def save_npz(out: Path, s: Session, session_id: int | None = None):
    arrs = dict(
        name=s.name,
        t_s=s.t_s,
        accel_phone=s.accel_phone,
        gyro=s.gyro,
        linacc=s.linacc,
        accel_enu=s.accel_enu,
        accel_veh=s.accel_veh,
        azimuth_deg=s.azimuth_deg,
        pitch_deg=s.pitch_deg,
        roll_deg=s.roll_deg,
        heading_deg=s.heading_deg,
        speed_ms=s.speed_ms,
        speed_acc_ms=s.speed_acc_ms,
        hacc_m=s.hacc_m,
        valid_label=s.valid_label,
        in_gap=s.in_gap,
        stationary=s.stationary,
        spoof_flag=s.spoof_flag,
        g_phone=s.g_phone,
        target_hz=np.float32(TARGET_HZ),
    )
    if session_id is not None:
        arrs['session_id'] = np.full_like(s.t_s, session_id, dtype=np.int16)
    np.savez_compressed(out, **arrs)
    print(f"  wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


def merge_and_save(sessions: list[Session], out_path: Path):
    if not sessions:
        return
    arrays = {}
    keys = ['t_s','accel_phone','gyro','linacc','accel_enu','accel_veh',
            'azimuth_deg','pitch_deg','roll_deg','heading_deg',
            'speed_ms','speed_acc_ms','hacc_m',
            'valid_label','in_gap','stationary','spoof_flag']
    for k in keys:
        arrays[k] = np.concatenate([getattr(s, k) for s in sessions], axis=0)
    session_id = np.concatenate(
        [np.full_like(s.t_s, i, dtype=np.int16) for i, s in enumerate(sessions)]
    )
    arrays['session_id']    = session_id
    arrays['session_names'] = np.array([s.name for s in sessions])
    arrays['target_hz']     = np.float32(TARGET_HZ)
    np.savez_compressed(out_path, **arrays)
    n = arrays['t_s'].shape[0]
    valid_pct = 100.0 * arrays['valid_label'].mean()
    print(f"\n=== merged ===\n  samples={n}  duration={n*DT/60:.1f} min  valid_label={valid_pct:.1f}%")
    print(f"  wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default=r'E:\Project\cnn_speed_detector\samples')
    ap.add_argument('--out',     default=r'E:\Project\cnn_speed_detector\samples\preprocessed')
    ap.add_argument('--no-merge', action='store_true')
    args = ap.parse_args()

    samples_root = Path(args.samples)
    out_root     = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    folders = sorted([p for p in samples_root.iterdir() if p.is_dir() and p.name.startswith('navlog_')])
    if not folders:
        print(f"No navlog_* folders in {samples_root}", file=sys.stderr); sys.exit(1)

    sessions: list[Session] = []
    for i, folder in enumerate(folders):
        try:
            s = preprocess_session(folder)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            continue
        save_npz(out_root / f"{folder.name}.npz", s, session_id=i)
        sessions.append(s)

    if not args.no_merge and sessions:
        merge_and_save(sessions, out_root / "merged.npz")


if __name__ == '__main__':
    main()
