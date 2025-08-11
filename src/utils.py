from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np

# Try to import mediapipe for native drawing support (only available in Py<=3.12)
try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None  # on Python 3.13 this will be None


def extract_features(landmarks_norm: np.ndarray | object, frame_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Convert 21 hand landmarks (normalized, wrist-centered) to a feature vector.
    - Translate so wrist (landmark 0) is origin.
    - Scale by max pairwise XY distance to make scale-invariant.
    - Return flattened (x,y,z) for 21 points -> 63-dim vector.
    """
    # Accept either MediaPipe landmark list or numpy (21,3)
    if mp is not None and hasattr(landmarks_norm, "landmark"):
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        for lm in landmarks_norm.landmark:  # type: ignore[attr-defined]
            xs.append(lm.x)
            ys.append(lm.y)
            zs.append(lm.z)
        pts = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    else:
        pts = np.asarray(landmarks_norm, dtype=np.float32)
        if pts.shape != (21, 3):
            pts = pts.reshape(-1, 3)
            if pts.shape[0] != 21:
                raise ValueError("Expected 21x3 landmarks array")

    # Translate: subtract wrist
    origin = pts[0:1, :]  # wrist is index 0
    rel = pts - origin

    # Scale: use max XY distance among points
    xy = rel[:, :2]
    # Compute pairwise distances efficiently
    diffs = xy[:, None, :] - xy[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    scale = float(np.max(dists))
    if scale < 1e-6:
        scale = 1.0
    rel /= scale

    # Flatten
    feats = rel.reshape(-1)
    return feats.astype(np.float32)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20)  # pinky
]

def draw_hand_landmarks(frame: np.ndarray, landmarks_norm: np.ndarray | object, color=(0,255,0)) -> None:
    """Draw landmarks.
    - If MediaPipe is available and input is a MediaPipe landmark object, use mp's drawing for better visuals.
    - Else, assume numpy (21,3) in normalized coords and draw with OpenCV.
    """
    if mp is not None and hasattr(landmarks_norm, "landmark"):
        mp_drawing = mp.solutions.drawing_utils  # type: ignore
        mp_style = mp.solutions.drawing_styles   # type: ignore
        mp_drawing.draw_landmarks(
            frame,
            landmarks_norm,  # type: ignore[arg-type]
            mp.solutions.hands.HAND_CONNECTIONS,  # type: ignore
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style(),
        )
        return

    h, w = frame.shape[:2]
    pts = np.asarray(landmarks_norm, dtype=np.float32)
    if pts.shape != (21, 3):
        return
    # landmarks_norm are assumed in normalized image coords [0,1]
    # convert to pixel coordinates
    xy = pts[:, :2]
    xy_px = np.stack([xy[:,0] * w, xy[:,1] * h], axis=1).astype(int)

    # draw connections
    for a, b in HAND_CONNECTIONS:
        pa = tuple(xy_px[a])
        pb = tuple(xy_px[b])
        cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)

    # draw points
    for p in xy_px:
        cv2.circle(frame, tuple(p), 3, (255, 0, 0), -1, cv2.LINE_AA)


def put_text_box(img: np.ndarray, lines: Iterable[str], org: Tuple[int, int], font_scale: float = 0.6, thickness: int = 1) -> None:
    """Draw a semi-transparent text box with multiple lines starting at org (top-left)."""
    x, y = org
    pad = 8
    line_h = int(18 * font_scale) + 6
    width = 0
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        width = max(width, w)
    height = line_h * len(list(lines)) if not isinstance(lines, list) else line_h * len(lines)

    # background rectangle
    bg_tl = (x - pad, y - pad)
    bg_br = (x + width + pad, y + height + pad)
    overlay = img.copy()
    cv2.rectangle(overlay, bg_tl, bg_br, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # text
    y_text = y + int(line_h * 0.7)
    for line in lines:
        cv2.putText(img, line, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_text += line_h
