from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


class OnnxHandTracker:
    """
    ONNX-based hand tracker interface.

    NOTE: This class expects pre-converted ONNX models compatible with the
    MediaPipe Hands pipeline:
      - Palm detection model (palm_detection.onnx)
      - Hand landmark model (hand_landmark.onnx)

    Implementing the full MediaPipe decoding (anchors, NMS, ROI cropping,
    normalization to 21 landmarks) is non-trivial and out of scope here. This
    module is a scaffold that will return None until valid models and the
    decoding steps are provided.

    You can supply your own models at:
      models/palm_detection.onnx
      models/hand_landmark.onnx

    If provided, you can extend `_infer_palm` and `_infer_landmarks` to complete
    the pipeline. For now, `process()` will return None without crashing, so the
    rest of the app remains usable (e.g., UI, TTS).
    """

    def __init__(self, models_dir: Optional[str | Path] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.palm_path = self.models_dir / "palm_detection.onnx"
        self.lmk_path = self.models_dir / "hand_landmark.onnx"

        self._warned = False

        self.palm_sess = None
        self.lmk_sess = None

        self._try_load_sessions()

    def _try_load_sessions(self) -> None:
        if ort is None:
            return
        try:
            if self.palm_path.exists():
                self.palm_sess = ort.InferenceSession(str(self.palm_path), providers=["CPUExecutionProvider"]) 
            if self.lmk_path.exists():
                self.lmk_sess = ort.InferenceSession(str(self.lmk_path), providers=["CPUExecutionProvider"]) 
        except Exception:
            self.palm_sess = None
            self.lmk_sess = None

    def process(self, bgr_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns normalized 21x3 landmarks (x,y in [0,1], z relative) or None.
        Currently returns None until proper ONNX models and decoding are supplied.
        """
        if ort is None or self.palm_sess is None or self.lmk_sess is None:
            if not self._warned:
                print(
                    "ONNX hand models not found or onnxruntime not installed. "
                    "Place palm_detection.onnx and hand_landmark.onnx under models/ "
                    "or install onnxruntime per requirements.txt."
                )
                self._warned = True
            return None

        # Placeholder pipeline: not implemented.
        # To implement, you would:
        # 1) Preprocess frame -> input for palm model, run palm detection, decode boxes.
        # 2) Select best ROI, crop/warp hand region, run landmark model.
        # 3) Postprocess to 21 3D landmarks normalized to image size.
        # For now, return None so UI remains responsive.
        return None
