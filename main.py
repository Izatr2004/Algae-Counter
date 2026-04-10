#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import os

try:
    import modal
except Exception:
    modal = None

import cv2
import numpy as np


try:
    from cellpose import models as cellpose_models
except Exception:
    cellpose_models = None

IMAGE_EXTS = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}


@dataclass
class Config:
    detector: str = 'threshold'
    use_cellpose_gpu: bool = False
    cellpose_model_type: str = 'cyto3'
    cellpose_diameter: Optional[float] = None
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = -1.0
    cellpose_min_size_px: int = 5
    cellpose_blackhat_px: int = 21
    cellpose_normalize: bool = True
    cellpose_batch_size: int = 1
    cellpose_tile_size: int = 256
    verbose: bool = True

    bright_quantile: float = 99.0
    projection_threshold_ratio: float = 0.15
    projection_smooth_window: int = 25
    projection_group_gap: int = 60
    min_grid_spacing_px: int = 300

    background_blur_sigma: float = 15.0
    dark_threshold: float = 22.0
    grid_mask_gray_threshold: int = 225
    grid_mask_dilate_px: int = 9
    blob_open_px: int = 3

    min_blob_area_frac: float = 2.5e-5
    max_blob_area_frac_abs: float = 4.0e-4
    max_blob_aspect_ratio: float = 4.0

    grid_line_margin_frac: float = 0.04
    merge_radius_frac: float = 0.05
    debug_canvas_px_per_cell: int = 260


@dataclass
class ImageInfo:
    file_path: str
    sample_key: str
    image: np.ndarray
    x_lines: List[float]
    y_lines: List[float]
    x0: float
    sx: float
    y_candidates: List[Tuple[float, float, float, Tuple[int, ...]]]
    align_points_xy: np.ndarray


@dataclass
class BlobDetection:
    sample_key: str
    file_name: str
    source_index: int
    cx: float
    cy: float
    u: float
    v: float
    row: int
    col: int
    area_px: int
    area_frac: float
    bbox_w: int
    bbox_h: int


@dataclass
class ClusteredDetection:
    sample_key: str
    row: int
    col: int
    u: float
    v: float
    n_supporting_detections: int
    median_area_frac: float


class _CellposeCache:
    model = None
    key = None


def sample_key_from_stem(stem: str) -> str:
    # "ABCD 1 (2)" -> "ABCD 1"
    m = re.match(r"^(.*?)\s*\((\d+)\)\s*$", stem)
    if m:
        return m.group(1).strip()
    return stem.strip()


def list_images(input_dir: Path) -> List[Path]:
    files = []
    for p in input_dir.iterdir():
        if not (p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            continue
        # Keep only files whose stem ends with a parenthesized photo index, e.g. 'ABCD 1 (2)'.
        if re.match(r"^.*\(\d+\)\s*$", p.stem):
            files.append(p)
    return sorted(files)


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode='same')


def line_centers_1d(projection: np.ndarray, threshold_ratio: float, smooth_window: int, group_gap: int) -> List[float]:
    sm = smooth_1d(projection, smooth_window)
    if sm.max() <= 0:
        return []
    threshold = float(sm.max() * threshold_ratio)
    idx = np.where(sm >= threshold)[0]
    if len(idx) == 0:
        return []

    groups: List[Tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev <= group_gap:
            prev = i
        else:
            groups.append((start, prev))
            start = prev = i
    groups.append((start, prev))

    centers: List[float] = []
    for a, b in groups:
        r = np.arange(a, b + 1, dtype=np.float32)
        w = sm[a:b + 1]
        if w.sum() <= 0:
            continue
        centers.append(float((r * w).sum() / w.sum()))
    return centers


def detect_line_candidates(image: np.ndarray, cfg: Config) -> Tuple[List[float], List[float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bright_thresh = max(1.0, float(np.percentile(gray, cfg.bright_quantile)))
    bright_mask = (gray >= bright_thresh).astype(np.uint8)

    x_projection = bright_mask.sum(axis=0)
    y_projection = bright_mask.sum(axis=1)
    x_lines = line_centers_1d(x_projection, cfg.projection_threshold_ratio, cfg.projection_smooth_window, cfg.projection_group_gap)
    y_lines = line_centers_1d(y_projection, cfg.projection_threshold_ratio, cfg.projection_smooth_window, cfg.projection_group_gap)
    return x_lines, y_lines


def best_equal_subset(points: Sequence[float], min_len: int, max_len: int, min_spacing: float) -> List[float]:
    pts = list(points)
    if len(pts) <= min_len:
        return pts

    best_score = None
    best_subset = pts[:]
    for k in range(min_len, min(max_len, len(pts)) + 1):
        for idxs in itertools.combinations(range(len(pts)), k):
            sub = [pts[i] for i in idxs]
            diffs = np.diff(sub)
            if len(diffs) == 0 or float(np.min(diffs)) < min_spacing:
                continue
            spacing = float(np.median(diffs))
            rel_err = float(np.mean(np.abs(diffs - spacing)) / max(spacing, 1e-9))
            span = float(sub[-1] - sub[0])
            score = (k, -rel_err, span)
            if best_score is None or score > best_score:
                best_score = score
                best_subset = sub
    return best_subset




def infer_regular_grid_lines(
    points: Sequence[float],
    required_count: int,
    image_extent: int,
    min_spacing: float,
    spacing_hint: Optional[float] = None,
) -> List[float]:
    pts = sorted(float(v) for v in points)
    if len(pts) >= required_count:
        sub = best_equal_subset(pts, min_len=required_count, max_len=required_count, min_spacing=min_spacing)
        if len(sub) == required_count:
            return [float(v) for v in sub]

    spacing_candidates: List[float] = []
    if spacing_hint is not None and float(spacing_hint) > 0:
        spacing_candidates.append(float(spacing_hint))
    if len(pts) >= 2:
        diffs = np.diff(np.asarray(pts, dtype=np.float32))
        for d in diffs:
            d = float(d)
            for step in range(1, required_count):
                s = d / step
                if s >= 0.55 * min_spacing:
                    spacing_candidates.append(float(s))
    if not spacing_candidates:
        return [float(v) for v in pts]

    # Deduplicate near-identical spacing guesses.
    spacing_candidates = sorted(spacing_candidates)
    uniq_spacings: List[float] = []
    for s in spacing_candidates:
        if not uniq_spacings or abs(s - uniq_spacings[-1]) > max(12.0, 0.03 * s):
            uniq_spacings.append(s)

    best_lines: List[float] = []
    best_score: Optional[Tuple[float, float, float, float]] = None
    anchor_points = pts if pts else [0.5 * (image_extent - 1)]
    for s in uniq_spacings:
        tol = max(20.0, 0.08 * s)
        for p in anchor_points:
            for idx in range(required_count):
                start = p - idx * s
                lines = [start + j * s for j in range(required_count)]
                dists = [min(abs(q - l) for l in lines) for q in pts] if pts else []
                support = sum(d <= tol for d in dists)
                mae = float(np.mean(dists)) if dists else 0.0
                in_bounds = sum((-0.15 * s) <= l <= (image_extent - 1 + 0.15 * s) for l in lines)
                near_edge_bonus = 0
                if any(abs(l) <= 0.22 * s for l in lines):
                    near_edge_bonus += 1
                if any(abs(l - (image_extent - 1)) <= 0.22 * s for l in lines):
                    near_edge_bonus += 1
                centered_penalty = abs((lines[0] + lines[-1]) * 0.5 - (image_extent - 1) * 0.5)
                spacing_match = 0.0
                if spacing_hint is not None and float(spacing_hint) > 0:
                    spacing_match = -abs(s - float(spacing_hint)) / float(spacing_hint)
                score = (support, spacing_match, in_bounds, near_edge_bonus, -mae - 0.001 * centered_penalty)
                if best_score is None or score > best_score:
                    best_score = score
                    best_lines = [float(v) for v in lines]

    if not best_lines:
        return [float(v) for v in pts]
    return best_lines

def fit_line_mapping(points: Sequence[float], indices: Sequence[int]) -> Tuple[float, float, float]:
    p = np.asarray(points, dtype=np.float32)
    k = np.asarray(indices, dtype=np.float32)
    A = np.vstack([np.ones_like(k), k]).T
    coeff, *_ = np.linalg.lstsq(A, p, rcond=None)
    y0, spacing = float(coeff[0]), float(coeff[1])
    pred = y0 + spacing * k
    err = float(np.mean(np.abs(pred - p))) if len(p) else 0.0
    return y0, spacing, err


def build_image_info(path: Path, cfg: Config) -> ImageInfo:
    if cfg.verbose:
        print(f'[Grid] analyzing {path.name}')
    image = load_rgb(path)
    h, w = image.shape[:2]
    x_lines_raw, y_lines_raw = detect_line_candidates(image, cfg)

    x_lines = infer_regular_grid_lines(
        x_lines_raw,
        required_count=5,
        image_extent=w,
        min_spacing=cfg.min_grid_spacing_px,
        spacing_hint=None,
    )
    if len(x_lines) != 5:
        raise ValueError(
            f"Could not infer 5 vertical grid lines in {path.name}. Detected raw candidates: {x_lines_raw}"
        )

    x0, sx, _ = fit_line_mapping(x_lines, [0, 1, 2, 3, 4])

    y_lines = infer_regular_grid_lines(
        y_lines_raw,
        required_count=5,
        image_extent=h,
        min_spacing=cfg.min_grid_spacing_px,
        spacing_hint=sx,
    )
    if len(y_lines) != 5:
        raise ValueError(
            f"Could not infer 5 horizontal grid lines in {path.name}. Detected raw candidates: {y_lines_raw}"
        )

    y0, sy, yerr = fit_line_mapping(y_lines, [0, 1, 2, 3, 4])
    y_candidates: List[Tuple[float, float, float, Tuple[int, ...]]] = [(y0, sy, yerr, (0, 1, 2, 3, 4))]

    if cfg.verbose and (len(x_lines_raw) != 5 or len(y_lines_raw) < 3):
        print(
            f'[Grid] fallback inference used for {path.name}: '
            f'raw_x={list(map(lambda v: round(float(v),1), x_lines_raw))}, '
            f'raw_y={list(map(lambda v: round(float(v),1), y_lines_raw))}, '
            f'inferred_x={list(map(lambda v: round(float(v),1), x_lines))}, '
            f'inferred_y={list(map(lambda v: round(float(v),1), y_lines))}'
        )

    align_points_xy = detect_blobs_for_alignment(image, sx, sy, cfg)

    return ImageInfo(
        file_path=str(path),
        sample_key=sample_key_from_stem(path.stem),
        image=image,
        x_lines=list(map(float, x_lines)),
        y_lines=list(map(float, y_lines)),
        x0=float(x0),
        sx=float(sx),
        y_candidates=y_candidates,
        align_points_xy=align_points_xy,
    )


def morph_disk(diameter: int) -> np.ndarray:
    diameter = max(1, int(diameter))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def get_cellpose_model(cfg: Config):
    if cellpose_models is None:
        raise ImportError(
            "Cellpose is not installed. Install it with `pip install cellpose` and rerun with --detector cellpose."
        )

    pretrained_model = cfg.cellpose_model_type
    if pretrained_model is None or str(pretrained_model).strip() == '':
        pretrained_model = 'cpsam'
    pretrained_model = str(pretrained_model)

    key = (cfg.use_cellpose_gpu, pretrained_model)
    if _CellposeCache.model is None or _CellposeCache.key != key:
        if cfg.verbose:
            print(f'[Cellpose] loading model: {pretrained_model} (gpu={cfg.use_cellpose_gpu})')
        # In Cellpose v4+, CellposeModel expects pretrained_model=... .
        # Passing model_type=... triggers a warning and can ignore the requested model.
        _CellposeCache.model = cellpose_models.CellposeModel(
            gpu=cfg.use_cellpose_gpu,
            pretrained_model=pretrained_model,
        )
        _CellposeCache.key = key
        if cfg.verbose:
            print('[Cellpose] model loaded')
    return _CellposeCache.model


def preprocess_for_cellpose(image: np.ndarray, cfg: Config) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if cfg.cellpose_blackhat_px > 1:
        k = morph_disk(cfg.cellpose_blackhat_px)
        enhanced = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    else:
        enhanced = gray

    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    rgb = np.repeat(enhanced[:, :, None], 3, axis=2)

    grid_mask = (gray > cfg.grid_mask_gray_threshold).astype(np.uint8) * 255
    grid_mask = cv2.dilate(grid_mask, morph_disk(cfg.grid_mask_dilate_px))
    rgb[grid_mask > 0] = 0
    return rgb


def masks_to_blob_dicts(masks: np.ndarray, sx: float, sy: float, cfg: Config) -> List[Dict[str, float]]:
    cell_area = max(sx * sy, 1.0)
    blobs: List[Dict[str, float]] = []
    for label in np.unique(masks):
        if int(label) == 0:
            continue
        ys, xs = np.where(masks == label)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        area = int(len(xs))
        area_frac = float(area) / cell_area
        aspect = float(max(w, h)) / max(1.0, float(min(w, h)))
        if area_frac < cfg.min_blob_area_frac or area_frac > cfg.max_blob_area_frac_abs:
            continue
        if aspect > cfg.max_blob_aspect_ratio:
            continue
        blobs.append(
            {
                'cx': float(xs.mean()),
                'cy': float(ys.mean()),
                'area_px': int(area),
                'area_frac': float(area_frac),
                'bbox_w': int(w),
                'bbox_h': int(h),
            }
        )
    return blobs


def detect_blobs_cellpose(image: np.ndarray, sx: float, sy: float, cfg: Config, context: str = '') -> List[Dict[str, float]]:
    model = get_cellpose_model(cfg)
    cp_input = preprocess_for_cellpose(image, cfg)
    if cfg.verbose:
        shape = 'x'.join(map(str, cp_input.shape[:2]))
        label = f' for {context}' if context else ''
        print(f'[Cellpose] running{label} on image {shape}')
    masks, flows, styles = model.eval(
        cp_input,
        channel_axis=2,
        normalize=cfg.cellpose_normalize,
        diameter=cfg.cellpose_diameter,
        flow_threshold=cfg.cellpose_flow_threshold,
        cellprob_threshold=cfg.cellpose_cellprob_threshold,
        min_size=cfg.cellpose_min_size_px,
        batch_size=max(1, int(cfg.cellpose_batch_size)),
        bsize=max(64, int(cfg.cellpose_tile_size)),
    )
    blobs = masks_to_blob_dicts(np.asarray(masks), sx, sy, cfg)
    if cfg.verbose:
        label = f' for {context}' if context else ''
        print(f'[Cellpose] kept {len(blobs)} candidate blobs{label}')
    return blobs


def dark_blob_mask(image: np.ndarray, cfg: Config) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), cfg.background_blur_sigma)
    dark = np.clip(bg - gray, 0, None)

    grid_mask = (gray > cfg.grid_mask_gray_threshold).astype(np.uint8) * 255
    grid_mask = cv2.dilate(grid_mask, morph_disk(cfg.grid_mask_dilate_px))

    mask = (dark > cfg.dark_threshold).astype(np.uint8) * 255
    mask[grid_mask > 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_disk(cfg.blob_open_px))
    return mask


def detect_blobs_for_alignment(image: np.ndarray, sx: float, sy_guess: float, cfg: Config) -> np.ndarray:
    if cfg.detector == 'cellpose':
        blobs = detect_blobs_cellpose(image, sx, sy_guess, cfg, context='alignment')
        pts = [[float(b['cx']), float(b['cy'])] for b in blobs]
        return np.asarray(pts, dtype=np.float32)

    mask = dark_blob_mask(image, cfg)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    cell_area = max(sx * sy_guess, 1.0)

    pts: List[List[float]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        area_frac = float(area) / cell_area
        aspect = float(max(w, h)) / max(1.0, float(min(w, h)))
        if area_frac < cfg.min_blob_area_frac or area_frac > cfg.max_blob_area_frac_abs:
            continue
        if aspect > cfg.max_blob_aspect_ratio:
            continue
        pts.append([float(cents[i, 0]), float(cents[i, 1])])
    return np.asarray(pts, dtype=np.float32)


def map_points_to_grid(points_xy: np.ndarray, x0: float, sx: float, y0: float, sy: float) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    uv = np.empty_like(points_xy, dtype=np.float32)
    uv[:, 0] = (points_xy[:, 0] - x0) / sx
    uv[:, 1] = (points_xy[:, 1] - y0) / sy
    return uv


def duplicate_score(mapped_point_sets: Sequence[np.ndarray], radius_frac: float) -> int:
    total = 0
    radius2 = radius_frac * radius_frac
    for i in range(len(mapped_point_sets)):
        for j in range(i + 1, len(mapped_point_sets)):
            a = mapped_point_sets[i]
            b = mapped_point_sets[j]
            if len(a) == 0 or len(b) == 0:
                continue

            a_in = (a[:, 0] > -0.1) & (a[:, 0] < 4.1) & (a[:, 1] > -0.1) & (a[:, 1] < 4.1)
            b_in = (b[:, 0] > -0.1) & (b[:, 0] < 4.1) & (b[:, 1] > -0.1) & (b[:, 1] < 4.1)
            a = a[a_in]
            b = b[b_in]
            if len(a) == 0 or len(b) == 0:
                continue

            d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
            total += int((d2.min(axis=1) < radius2).sum())
            total += int((d2.min(axis=0) < radius2).sum())
    return total


def choose_best_mappings(infos: Sequence[ImageInfo], cfg: Config) -> List[Tuple[float, float, float, float, Tuple[int, ...]]]:
    candidate_lists: List[List[Tuple[float, float, float, float, Tuple[int, ...]]]] = []
    for info in infos:
        cands: List[Tuple[float, float, float, float, Tuple[int, ...]]] = []
        for y0, sy, fit_err, idxs in info.y_candidates:
            cands.append((info.x0, info.sx, y0, sy, idxs))
        candidate_lists.append(cands)

    best_combo = None
    best_score = None
    for combo in itertools.product(*candidate_lists):
        mapped = [map_points_to_grid(info.align_points_xy, x0, sx, y0, sy) for info, (x0, sx, y0, sy, _) in zip(infos, combo)]
        score = duplicate_score(mapped, cfg.merge_radius_frac)
        if best_score is None or score > best_score:
            best_score = score
            best_combo = combo

    assert best_combo is not None
    return list(best_combo)


def detect_blobs(image: np.ndarray, sx: float, sy: float, cfg: Config) -> List[Dict[str, float]]:
    if cfg.detector == 'cellpose':
        return detect_blobs_cellpose(image, sx, sy, cfg, context='final detection')

    mask = dark_blob_mask(image, cfg)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    cell_area = max(sx * sy, 1.0)

    out: List[Dict[str, float]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        area_frac = float(area) / cell_area
        aspect = float(max(w, h)) / max(1.0, float(min(w, h)))
        if area_frac < cfg.min_blob_area_frac or area_frac > cfg.max_blob_area_frac_abs:
            continue
        if aspect > cfg.max_blob_aspect_ratio:
            continue
        out.append(
            {
                'cx': float(cents[i, 0]),
                'cy': float(cents[i, 1]),
                'area_px': int(area),
                'area_frac': float(area_frac),
                'bbox_w': int(w),
                'bbox_h': int(h),
            }
        )
    return out


def too_close_to_grid_line(u: float, v: float, margin: float) -> bool:
    du = min(abs(u - round(u)), abs((u % 1.0) - 1.0), abs(u % 1.0))
    dv = min(abs(v - round(v)), abs((v % 1.0) - 1.0), abs(v % 1.0))
    frac_u = min(u % 1.0, 1.0 - (u % 1.0))
    frac_v = min(v % 1.0, 1.0 - (v % 1.0))
    return frac_u < margin or frac_v < margin


def collect_detections(sample_key: str, infos: Sequence[ImageInfo], mappings: Sequence[Tuple[float, float, float, float, Tuple[int, ...]]], cfg: Config) -> List[BlobDetection]:
    detections: List[BlobDetection] = []
    for source_index, (info, (x0, sx, y0, sy, _)) in enumerate(zip(infos, mappings)):
        blobs = detect_blobs(info.image, sx, sy, cfg)
        for b in blobs:
            u = (b['cx'] - x0) / sx
            v = (b['cy'] - y0) / sy
            if not (0.0 < u < 4.0 and 0.0 < v < 4.0):
                continue
            if too_close_to_grid_line(u, v, cfg.grid_line_margin_frac):
                continue
            row = int(v)
            col = int(u)
            detections.append(
                BlobDetection(
                    sample_key=sample_key,
                    file_name=Path(info.file_path).name,
                    source_index=source_index,
                    cx=float(b['cx']),
                    cy=float(b['cy']),
                    u=float(u),
                    v=float(v),
                    row=row,
                    col=col,
                    area_px=int(b['area_px']),
                    area_frac=float(b['area_frac']),
                    bbox_w=int(b['bbox_w']),
                    bbox_h=int(b['bbox_h']),
                )
            )
    return detections


def cluster_detections(detections: Sequence[BlobDetection], cfg: Config) -> List[ClusteredDetection]:
    clusters: List[Dict[str, object]] = []
    r2 = cfg.merge_radius_frac * cfg.merge_radius_frac
    for det in detections:
        best_idx = None
        best_d2 = None
        for i, c in enumerate(clusters):
            du = det.u - float(c['u'])
            dv = det.v - float(c['v'])
            d2 = du * du + dv * dv
            if d2 <= r2 and (best_d2 is None or d2 < best_d2):
                best_idx = i
                best_d2 = d2
        if best_idx is None:
            clusters.append({'u': det.u, 'v': det.v, 'members': [det]})
        else:
            members: List[BlobDetection] = clusters[best_idx]['members']  # type: ignore[assignment]
            members.append(det)
            clusters[best_idx]['u'] = float(np.mean([m.u for m in members]))
            clusters[best_idx]['v'] = float(np.mean([m.v for m in members]))

    merged: List[ClusteredDetection] = []
    for c in clusters:
        members: List[BlobDetection] = c['members']  # type: ignore[assignment]
        u = float(c['u'])
        v = float(c['v'])
        if not (0.0 <= u < 4.0 and 0.0 <= v < 4.0):
            continue
        row = min(3, max(0, int(v)))
        col = min(3, max(0, int(u)))
        merged.append(
            ClusteredDetection(
                sample_key=members[0].sample_key,
                row=row,
                col=col,
                u=u,
                v=v,
                n_supporting_detections=len(members),
                median_area_frac=float(np.median([m.area_frac for m in members])),
            )
        )
    return merged


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_debug_canvas(sample_key: str, detections: Sequence[ClusteredDetection], output_path: Path, cfg: Config) -> None:
    cell = cfg.debug_canvas_px_per_cell
    margin = 30
    h = 4 * cell + 2 * margin
    w = 4 * cell + 2 * margin
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)

    for i in range(5):
        y = margin + i * cell
        x = margin + i * cell
        cv2.line(canvas, (margin, y), (margin + 4 * cell, y), (200, 200, 200), 2)
        cv2.line(canvas, (x, margin), (x, margin + 4 * cell), (200, 200, 200), 2)

    for r in range(4):
        for c in range(4):
            cv2.putText(canvas, f"{r},{c}", (margin + c * cell + 8, margin + r * cell + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)

    for det in detections:
        x = int(round(margin + det.u * cell))
        y = int(round(margin + det.v * cell))
        radius = 3 if det.n_supporting_detections == 1 else 4
        color = (40, 120, 30) if det.n_supporting_detections == 1 else (20, 60, 180)
        cv2.circle(canvas, (x, y), radius, color, -1)

    cv2.putText(canvas, f"{sample_key}: {len(detections)} algae", (margin, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def process_sample(sample_key: str, paths: Sequence[Path], output_dir: Path, cfg: Config) -> Dict[str, object]:
    if cfg.verbose:
        print(f'[Sample] processing {sample_key} with {len(paths)} image(s)')
    infos = [build_image_info(p, cfg) for p in sorted(paths)]
    mappings = choose_best_mappings(infos, cfg)
    detections = collect_detections(sample_key, infos, mappings, cfg)
    clusters = cluster_detections(detections, cfg)

    box_counts = np.zeros((4, 4), dtype=np.int32)
    for c in clusters:
        box_counts[c.row, c.col] += 1

    render_debug_canvas(sample_key, clusters, output_dir / 'debug' / f"{sample_key}_detections.png", cfg)

    return {
        'sample_key': sample_key,
        'n_images': len(paths),
        'total_algae_count': int(box_counts.sum()),
        'box_counts': box_counts,
        'raw_detections': detections,
        'clustered_detections': clusters,
        'mappings': [
            {
                'file_name': Path(info.file_path).name,
                'x0': float(x0),
                'sx': float(sx),
                'y0': float(y0),
                'sy': float(sy),
                'row_line_indices_used': list(idxs),
                'detected_x_lines_px': [float(v) for v in info.x_lines],
                'detected_y_lines_px': [float(v) for v in info.y_lines],
            }
            for info, (x0, sx, y0, sy, idxs) in zip(infos, mappings)
        ],
    }


def build_outputs(results: Sequence[Dict[str, object]], output_dir: Path, cfg: Config) -> None:
    sample_rows: List[Dict[str, object]] = []
    box_rows: List[Dict[str, object]] = []
    raw_rows: List[Dict[str, object]] = []
    cluster_rows: List[Dict[str, object]] = []
    mapping_rows: List[Dict[str, object]] = []

    for res in results:
        sample_key = str(res['sample_key'])
        box_counts = res['box_counts']
        sample_rows.append(
            {
                'sample_key': sample_key,
                'n_images': int(res['n_images']),
                'total_algae_count': int(res['total_algae_count']),
            }
        )
        for r in range(4):
            for c in range(4):
                box_rows.append(
                    {
                        'sample_key': sample_key,
                        'row': r,
                        'col': c,
                        'algae_count': int(box_counts[r, c]),
                    }
                )
        for det in res['raw_detections']:
            raw_rows.append(asdict(det))
        for det in res['clustered_detections']:
            cluster_rows.append(asdict(det))
        for mapping in res['mappings']:
            mapping_rows.append({'sample_key': sample_key, **mapping})

    write_csv(output_dir / 'sample_counts.csv', sample_rows, ['sample_key', 'n_images', 'total_algae_count'])
    write_csv(output_dir / 'box_counts.csv', box_rows, ['sample_key', 'row', 'col', 'algae_count'])
    write_csv(
        output_dir / 'raw_detections.csv',
        raw_rows,
        ['sample_key', 'file_name', 'source_index', 'cx', 'cy', 'u', 'v', 'row', 'col', 'area_px', 'area_frac', 'bbox_w', 'bbox_h'],
    )
    write_csv(
        output_dir / 'clustered_detections.csv',
        cluster_rows,
        ['sample_key', 'row', 'col', 'u', 'v', 'n_supporting_detections', 'median_area_frac'],
    )
    write_csv(
        output_dir / 'mappings.csv',
        mapping_rows,
        ['sample_key', 'file_name', 'x0', 'sx', 'y0', 'sy', 'row_line_indices_used', 'detected_x_lines_px', 'detected_y_lines_px'],
    )

    summary = {
        'config': asdict(cfg),
        'n_samples': len(results),
        'sample_totals': {str(r['sample_key']): int(r['total_algae_count']) for r in results},
    }
    with (output_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Count algae dots inside imperfect 4x4 grid photos.')
    parser.add_argument('--input_dir', type=Path, required=True, help='Directory containing the images.')
    parser.add_argument('--output_dir', type=Path, required=True, help='Directory where CSVs/debug files will be written.')
    parser.add_argument('--bright_quantile', type=float, default=99.0)
    parser.add_argument('--detector', choices=['threshold', 'cellpose'], default='threshold')
    parser.add_argument('--use_cellpose_gpu', action='store_true')
    parser.add_argument('--cellpose_model_type', type=str, default='cyto3', help='Cellpose pretrained model name or path, e.g. cyto3, cpsam, or a custom model path.')
    parser.add_argument('--cellpose_diameter', type=float, default=None)
    parser.add_argument('--cellpose_flow_threshold', type=float, default=0.4)
    parser.add_argument('--cellpose_cellprob_threshold', type=float, default=-1.0)
    parser.add_argument('--cellpose_min_size_px', type=int, default=5)
    parser.add_argument('--cellpose_blackhat_px', type=int, default=21)
    parser.add_argument('--cellpose_batch_size', type=int, default=1)
    parser.add_argument('--cellpose_tile_size', type=int, default=256)
    parser.add_argument('--quiet', action='store_true', help='Reduce progress logging.')
    parser.add_argument('--dark_threshold', type=float, default=22.0)
    parser.add_argument('--min_blob_area_frac', type=float, default=2.5e-5)
    parser.add_argument('--max_blob_area_frac_abs', type=float, default=4.0e-4)
    parser.add_argument('--grid_line_margin_frac', type=float, default=0.04)
    parser.add_argument('--merge_radius_frac', type=float, default=0.05)
    return parser.parse_args()


def make_config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        detector=args.detector,
        use_cellpose_gpu=args.use_cellpose_gpu,
        cellpose_model_type=args.cellpose_model_type,
        cellpose_diameter=args.cellpose_diameter,
        cellpose_flow_threshold=args.cellpose_flow_threshold,
        cellpose_cellprob_threshold=args.cellpose_cellprob_threshold,
        cellpose_min_size_px=args.cellpose_min_size_px,
        cellpose_blackhat_px=args.cellpose_blackhat_px,
        cellpose_batch_size=args.cellpose_batch_size,
        cellpose_tile_size=args.cellpose_tile_size,
        verbose=not args.quiet,
        bright_quantile=args.bright_quantile,
        dark_threshold=args.dark_threshold,
        min_blob_area_frac=args.min_blob_area_frac,
        max_blob_area_frac_abs=args.max_blob_area_frac_abs,
        grid_line_margin_frac=args.grid_line_margin_frac,
        merge_radius_frac=args.merge_radius_frac,
    )


def run_pipeline(input_dir: Path, output_dir: Path, cfg: Config) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_images(input_dir)
    if not files:
        raise SystemExit(f'No images found in {input_dir}')

    grouped: Dict[str, List[Path]] = {}
    for path in files:
        key = sample_key_from_stem(path.stem)
        grouped.setdefault(key, []).append(path)

    if cfg.verbose:
        print(f'[Start] found {len(files)} image(s) across {len(grouped)} sample(s)')

    results: List[Dict[str, object]] = []
    for sample_key, paths in sorted(grouped.items()):
        try:
            results.append(process_sample(sample_key, paths, output_dir, cfg))
            print(f'[OK] {sample_key}: {results[-1]["total_algae_count"]} algae from {len(paths)} image(s)')
        except Exception as e:
            print(f'[FAILED] {sample_key}: {e}')

    build_outputs(results, output_dir, cfg)
    print(f'Wrote outputs to: {output_dir}')


def main() -> None:
    args = parse_args()
    cfg = make_config_from_args(args)
    run_pipeline(args.input_dir, args.output_dir, cfg)


if modal is not None:
    app = modal.App('algae-counter')
    image = (
        modal.Image.debian_slim(python_version='3.11')
        .pip_install('numpy', 'opencv-python-headless', 'pandas', 'scipy', 'cellpose')
    )
    input_vol = modal.Volume.from_name('algae-input', create_if_missing=True)
    output_vol = modal.Volume.from_name('algae-output', create_if_missing=True)
    model_vol = modal.Volume.from_name('algae-model-cache', create_if_missing=True)

    @app.function(
        image=image,
        gpu='a10g',
        timeout=60 * 60 * 6,
        volumes={
            '/vol/input': input_vol,
            '/vol/output': output_vol,
            '/root/.cellpose': model_vol,
        },
    )
    def modal_run(
        input_subdir: str = 'data',
        output_subdir: str = 'output',
        detector: str = 'cellpose',
        use_cellpose_gpu: bool = True,
        cellpose_model_type: str = 'cpsam',
        cellpose_diameter: float | None = None,
        cellpose_flow_threshold: float = 0.4,
        cellpose_cellprob_threshold: float = -1.0,
        cellpose_min_size_px: int = 5,
        cellpose_blackhat_px: int = 21,
        cellpose_batch_size: int = 1,
        cellpose_tile_size: int = 256,
        quiet: bool = False,
        bright_quantile: float = 99.0,
        dark_threshold: float = 22.0,
        min_blob_area_frac: float = 2.5e-5,
        max_blob_area_frac_abs: float = 4.0e-4,
        grid_line_margin_frac: float = 0.04,
        merge_radius_frac: float = 0.05,
    ) -> str:
        cfg = Config(
            detector=detector,
            use_cellpose_gpu=use_cellpose_gpu,
            cellpose_model_type=cellpose_model_type,
            cellpose_diameter=cellpose_diameter,
            cellpose_flow_threshold=cellpose_flow_threshold,
            cellpose_cellprob_threshold=cellpose_cellprob_threshold,
            cellpose_min_size_px=cellpose_min_size_px,
            cellpose_blackhat_px=cellpose_blackhat_px,
            cellpose_batch_size=cellpose_batch_size,
            cellpose_tile_size=cellpose_tile_size,
            verbose=not quiet,
            bright_quantile=bright_quantile,
            dark_threshold=dark_threshold,
            min_blob_area_frac=min_blob_area_frac,
            max_blob_area_frac_abs=max_blob_area_frac_abs,
            grid_line_margin_frac=grid_line_margin_frac,
            merge_radius_frac=merge_radius_frac,
        )
        input_dir = Path('/vol/input') / input_subdir
        output_dir = Path('/vol/output') / output_subdir
        run_pipeline(input_dir, output_dir, cfg)
        output_vol.commit()
        model_vol.commit()
        return str(output_dir)

    @app.local_entrypoint()
    def modal_main(
        input_subdir: str = 'data',
        output_subdir: str = 'output',
        detector: str = 'cellpose',
        use_cellpose_gpu: bool = True,
        cellpose_model_type: str = 'cpsam',
        cellpose_diameter: float | None = None,
        cellpose_flow_threshold: float = 0.4,
        cellpose_cellprob_threshold: float = -1.0,
        cellpose_min_size_px: int = 5,
        cellpose_blackhat_px: int = 21,
        cellpose_batch_size: int = 1,
        cellpose_tile_size: int = 256,
        quiet: bool = False,
        bright_quantile: float = 99.0,
        dark_threshold: float = 22.0,
        min_blob_area_frac: float = 2.5e-5,
        max_blob_area_frac_abs: float = 4.0e-4,
        grid_line_margin_frac: float = 0.04,
        merge_radius_frac: float = 0.05,
    ):
        modal_run.remote(
            input_subdir=input_subdir,
            output_subdir=output_subdir,
            detector=detector,
            use_cellpose_gpu=use_cellpose_gpu,
            cellpose_model_type=cellpose_model_type,
            cellpose_diameter=cellpose_diameter,
            cellpose_flow_threshold=cellpose_flow_threshold,
            cellpose_cellprob_threshold=cellpose_cellprob_threshold,
            cellpose_min_size_px=cellpose_min_size_px,
            cellpose_blackhat_px=cellpose_blackhat_px,
            cellpose_batch_size=cellpose_batch_size,
            cellpose_tile_size=cellpose_tile_size,
            quiet=quiet,
            bright_quantile=bright_quantile,
            dark_threshold=dark_threshold,
            min_blob_area_frac=min_blob_area_frac,
            max_blob_area_frac_abs=max_blob_area_frac_abs,
            grid_line_margin_frac=grid_line_margin_frac,
            merge_radius_frac=merge_radius_frac,
        )


if __name__ == '__main__':
    main()
