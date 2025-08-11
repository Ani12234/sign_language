from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import joblib
except Exception:  # joblib optional until user adds model
    joblib = None


class GestureClassifier:
    """
    Wraps a scikit-learn compatible classifier.
    - Expects model saved with joblib.
    - Labels provided via a text file (one label per line), else uses class indices.
    - Applies probability smoothing over a sliding window to stabilize predictions.
    - When a model is not available, falls back to a simple rule-based recognizer for a few gestures.
    """

    def __init__(
        self,
        model_path: str | Path,
        labels_path: Optional[str | Path] = None,
        smoothing: int = 8,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.labels_path = Path(labels_path) if labels_path else None
        self.smoothing = max(1, int(smoothing))

        self.model = None
        self.labels: List[str] | None = None
        self._probs_window: deque[np.ndarray] = deque(maxlen=self.smoothing)

        self._try_load_model()
        self._try_load_labels()

    def _try_load_model(self) -> None:
        if self.model_path and self.model_path.exists() and joblib is not None:
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None
        else:
            self.model = None

    def _try_load_labels(self) -> None:
        if self.labels_path and self.labels_path.exists():
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels = [line.strip() for line in f if line.strip()]
            except Exception:
                self.labels = None
        else:
            self.labels = None

    def _index_to_label(self, idx: int) -> str:
        if self.labels and 0 <= idx < len(self.labels):
            return self.labels[idx]
        return str(idx)

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict label and probability for a single feature vector.
        If model is not available, uses a heuristic rule-based fallback.
        """
        if self.model is None:
            return self._heuristic_predict(features)

        try:
            X = np.asarray(features, dtype=np.float32).reshape(1, -1)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
            else:
                # Build pseudo-probabilities from decision function or labels
                if hasattr(self.model, "decision_function"):
                    scores = self.model.decision_function(X)
                    if scores.ndim == 1:
                        # binary classifier -> convert to 2-class probs
                        scores = np.stack([-scores, scores], axis=-1)
                    # softmax
                    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                    probs = (e / np.sum(e, axis=-1, keepdims=True))[0]
                else:
                    # fallback: predict -> one-hot like probability
                    pred = int(self.model.predict(X)[0])
                    n_classes = len(self.labels) if self.labels else max(pred + 1, 2)
                    probs = np.zeros(n_classes, dtype=np.float32)
                    probs[pred] = 1.0

            # Smooth probabilities
            self._probs_window.append(np.asarray(probs, dtype=np.float32))
            avg_probs = np.mean(np.stack(self._probs_window, axis=0), axis=0)

            idx = int(np.argmax(avg_probs))
            label = self._index_to_label(idx)
            conf = float(avg_probs[idx])
            return label, conf
        except Exception:
            return "Unknown", 0.0

    # ----------------------------
    # Heuristic fallback
    # ----------------------------
    def _heuristic_predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Simple pretrained-like rule-based recognizer using relative landmark features.
        Recognizes gestures:
          - Open (all fingers extended)
          - Fist (no fingers extended)
          - Point (index only)
          - OK (index and thumb touching, others extended)
          - Two (index + middle)
          - Three (index + middle + ring)
          - Four (all except thumb)
          - Thumb (thumb only extended)
          - Pinch (thumb-index pinch, others curled)

        Returns (label, confidence) where confidence is a rough score [0,1].
        """
        try:
            pts = np.asarray(features, dtype=np.float32).reshape(-1, 3)
            if pts.shape[0] < 21:
                return "Unknown", 0.0

            # Landmark indices per MediaPipe Hands
            WRIST = 0
            THUMB_TIP = 4
            INDEX_TIP = 8
            MIDDLE_TIP = 12
            RING_TIP = 16
            PINKY_TIP = 20

            # Use y (vertical) relative to wrist to infer extension (smaller y means higher on image if not flipped)
            # But features are normalized and wrist-centered; use vector length from wrist for openness.
            wrist = pts[WRIST]
            tips = pts[[THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]]
            dists = np.linalg.norm(tips - wrist, axis=1)

            # Angle-based extension check at PIP joints (except thumb)
            # For each finger: angle at PIP between (MCP->PIP) and (TIP->PIP). Large angle (~180) => extended.
            def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
                v1 = a - b
                v2 = c - b
                n1 = np.linalg.norm(v1) + 1e-6
                n2 = np.linalg.norm(v2) + 1e-6
                cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                return float(np.degrees(np.arccos(cosang)))

            # Finger joints indices
            INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
            MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
            RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
            PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

            angle_index = angle(pts[INDEX_MCP], pts[INDEX_PIP], pts[INDEX_TIP])
            angle_middle = angle(pts[MIDDLE_MCP], pts[MIDDLE_PIP], pts[MIDDLE_TIP])
            angle_ring = angle(pts[RING_MCP], pts[RING_PIP], pts[RING_TIP])
            angle_pinky = angle(pts[PINKY_MCP], pts[PINKY_PIP], pts[PINKY_TIP])

            # Determine extension based on angles
            # Extended if angle > 155, curled if angle < 120
            ext_index = angle_index > 155
            ext_middle = angle_middle > 155
            ext_ring = angle_ring > 155
            ext_pinky = angle_pinky > 155

            # Thumb extension heuristic: distance from wrist relative to others
            ext_thumb = dists[0] > 0.30

            extended = np.array([ext_thumb, ext_index, ext_middle, ext_ring, ext_pinky], dtype=bool)
            curled = np.array([
                dists[0] < 0.18,  # thumb close
                angle_index < 120,
                angle_middle < 120,
                angle_ring < 120,
                angle_pinky < 120,
            ], dtype=bool)

            ext_count = int(extended.sum())

            # OK detection: thumb-index distance small, others extended
            thumb_index_dist = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]))
            if thumb_index_dist < 0.10 and extended[2] and extended[3] and extended[4]:
                score = max(0.0, 1.0 - thumb_index_dist / 0.12)
                return "OK", min(1.0, 0.6 + 0.4 * score)

            # Open hand (aka Five)
            if ext_count >= 4 and not curled.any():
                conf = min(1.0, 0.5 + 0.1 * ext_count)
                return "Open", conf

            # Fist
            if int(curled.sum()) >= 4:
                conf = min(1.0, 0.5 + 0.1 * int(curled.sum()))
                return "Fist", conf

            # Point (index extended only)
            if extended[1] and not extended[2] and not extended[3] and not extended[4]:
                conf = 0.7
                return "Point", conf

            # Two (index + middle)
            if extended[1] and extended[2] and not extended[3] and not extended[4]:
                return "Two", 0.75

            # Three (index + middle + ring)
            if extended[1] and extended[2] and extended[3] and not extended[4]:
                return "Three", 0.75

            # Four (index + middle + ring + pinky; thumb curled)
            if not extended[0] and extended[1] and extended[2] and extended[3] and extended[4]:
                return "Four", 0.8

            # Thumb only
            if extended[0] and not extended[1] and not extended[2] and not extended[3] and not extended[4]:
                return "Thumb", 0.75

            # Pinch (thumb-index close, others curled)
            if thumb_index_dist < 0.08 and not extended[2] and not extended[3] and not extended[4]:
                score = max(0.0, 1.0 - thumb_index_dist / 0.08)
                return "Pinch", min(1.0, 0.6 + 0.4 * score)

            return "Unknown", 0.3
        except Exception:
            return "Unknown", 0.0
