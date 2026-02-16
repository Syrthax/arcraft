"""
Hand Tracking Module
====================
Wraps MediaPipe Hand Landmarker (Tasks API) for real-time detection.

Responsibilities:
  - Process BGR frames from OpenCV.
  - Detect up to N hands and extract key landmarks.
  - Convert normalised MediaPipe coordinates → screen-pixel coordinates.
  - Expose a clean dataclass API consumed by ``gestures.py``.

Key landmarks extracted per hand:
  - Index finger tip  (landmark 8)
  - Thumb tip         (landmark 4)
  - Middle finger tip (landmark 12)
  - Palm centre       (average of wrist + four MCP joints)
"""

from __future__ import annotations

import os
import time

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HandLandmarks:
    """Processed landmark data for a single detected hand."""

    # Normalised (0-1) xyz positions from MediaPipe
    index_tip: np.ndarray
    thumb_tip: np.ndarray
    middle_tip: np.ndarray
    palm_center: np.ndarray

    # All 21 landmarks as a list of (x, y, z) arrays
    all_landmarks: List[np.ndarray]

    handedness: str  # 'Left' or 'Right'

    # Screen-pixel (u, v) positions – filled by HandTracker
    index_tip_screen: np.ndarray = field(default_factory=lambda: np.zeros(2))
    thumb_tip_screen: np.ndarray = field(default_factory=lambda: np.zeros(2))
    middle_tip_screen: np.ndarray = field(default_factory=lambda: np.zeros(2))
    palm_center_screen: np.ndarray = field(default_factory=lambda: np.zeros(2))


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class HandTracker:
    """
    Real-time hand tracker using the MediaPipe Tasks Hand Landmarker.

    Requires the ``hand_landmarker.task`` model file in the working
    directory (downloaded automatically by the project setup).

    Parameters
    ----------
    max_hands : int
        Maximum number of hands to detect simultaneously.
    detection_confidence : float
        Minimum confidence for initial detection (0-1).
    tracking_confidence : float
        Minimum confidence for subsequent frame tracking (0-1).
    model_path : str
        Path to the ``.task`` model file.
    """

    # MediaPipe landmark indices (see MediaPipe hand landmark model docs)
    INDEX_TIP   = 8
    THUMB_TIP   = 4
    MIDDLE_TIP  = 12
    RING_TIP    = 16
    PINKY_TIP   = 20
    WRIST       = 0
    INDEX_MCP   = 5
    MIDDLE_MCP  = 9
    RING_MCP    = 13
    PINKY_MCP   = 17

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        model_path: str = "hand_landmarker.task",
    ):
        # Resolve model path relative to this file's directory
        if not os.path.isabs(model_path):
            base = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "Download it from: https://storage.googleapis.com/"
                "mediapipe-models/hand_landmarker/hand_landmarker/"
                "float16/latest/hand_landmarker.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

        # Monotonic timestamp counter for VIDEO mode (milliseconds)
        self._ts_ms: int = 0

        # Cached frame dimensions (updated each call to process_frame)
        self._frame_w: int = 640
        self._frame_h: int = 480

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Optional[List[HandLandmarks]]:
        """
        Run hand detection on a BGR frame.

        Returns
        -------
        list[HandLandmarks] | None
            One entry per detected hand, or *None* when no hands are found.
        """
        self._frame_h, self._frame_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps
        self._ts_ms += 33  # ~30 fps step
        result = self._landmarker.detect_for_video(mp_image, self._ts_ms)

        if not result.hand_landmarks:
            return None

        hands: List[HandLandmarks] = []
        for lm_list, hand_info in zip(result.hand_landmarks, result.handedness):
            hands.append(self._build_hand(lm_list, hand_info))
        return hands

    def draw_landmarks(self, frame: np.ndarray, hands: Optional[List[HandLandmarks]]) -> np.ndarray:
        """
        Draw key landmarks on *frame* (mutates in-place) for visual
        feedback.
        """
        if hands is None:
            return frame

        for hand in hands:
            # Index finger tip – green
            pt = tuple(hand.index_tip_screen.astype(int))
            cv2.circle(frame, pt, 8, (0, 255, 0), -1)
            cv2.circle(frame, pt, 10, (0, 200, 0), 2)

            # Thumb tip – blue
            pt = tuple(hand.thumb_tip_screen.astype(int))
            cv2.circle(frame, pt, 8, (255, 100, 0), -1)

            # Middle finger tip – red
            pt = tuple(hand.middle_tip_screen.astype(int))
            cv2.circle(frame, pt, 6, (0, 0, 255), -1)

            # Palm centre – white
            pt = tuple(hand.palm_center_screen.astype(int))
            cv2.circle(frame, pt, 6, (255, 255, 255), 2)

            # Line between index and thumb (pinch visual)
            idx = tuple(hand.index_tip_screen.astype(int))
            thm = tuple(hand.thumb_tip_screen.astype(int))
            cv2.line(frame, idx, thm, (0, 255, 255), 2)

        return frame

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return (width, height) of the last processed frame."""
        return self._frame_w, self._frame_h

    def release(self) -> None:
        """Free MediaPipe resources."""
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_hand(self, lm_list, hand_info) -> HandLandmarks:
        """Convert Tasks API landmark list into a *HandLandmarks* instance."""
        landmarks = [
            np.array([lm.x, lm.y, lm.z], dtype=np.float32)
            for lm in lm_list
        ]

        # Palm centre ≈ average of wrist + four MCP joints
        palm_center = np.mean(
            [
                landmarks[self.WRIST],
                landmarks[self.INDEX_MCP],
                landmarks[self.MIDDLE_MCP],
                landmarks[self.RING_MCP],
                landmarks[self.PINKY_MCP],
            ],
            axis=0,
        )

        label = hand_info[0].category_name if hand_info else "Unknown"

        hand = HandLandmarks(
            index_tip=landmarks[self.INDEX_TIP],
            thumb_tip=landmarks[self.THUMB_TIP],
            middle_tip=landmarks[self.MIDDLE_TIP],
            palm_center=palm_center,
            all_landmarks=landmarks,
            handedness=label,
        )

        # Fill screen-pixel positions
        hand.index_tip_screen   = self._to_screen(hand.index_tip)
        hand.thumb_tip_screen   = self._to_screen(hand.thumb_tip)
        hand.middle_tip_screen  = self._to_screen(hand.middle_tip)
        hand.palm_center_screen = self._to_screen(hand.palm_center)

        return hand

    def _to_screen(self, normalised: np.ndarray) -> np.ndarray:
        """Map normalised (0-1) coordinates to pixel coordinates."""
        return np.array(
            [normalised[0] * self._frame_w, normalised[1] * self._frame_h],
            dtype=np.float32,
        )
