"""
Gesture Detection Module
========================
Analyses hand landmark data to recognise discrete gestures.

Supported gestures
------------------
* **PINCH_PLACE** – thumb tip + index finger tip close together.
  Triggers cube placement at the pinch midpoint.

* **PINCH_DELETE** – thumb tip + middle finger tip close together while
  index finger is extended (disambiguates from place).
  Triggers cube deletion at the pinch midpoint.

* **OPEN_PALM** – all five fingers spread away from the palm centre.
  Enables camera-orbit mode while held.

Detection logic
~~~~~~~~~~~~~~~
Each gesture uses Euclidean distance between normalised (0-1) landmark
positions.  A *cooldown timer* prevents the same gesture from firing
repeatedly while held.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from hand_tracking import HandLandmarks


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class GestureType(Enum):
    """Enumeration of all recognised gestures."""
    NONE         = auto()
    PINCH_PLACE  = auto()   # Thumb + index → place cube
    PINCH_DELETE = auto()   # Thumb + middle → delete cube
    OPEN_PALM    = auto()   # All fingers spread → camera mode


@dataclass
class GestureState:
    """Snapshot of a detected gesture with spatial metadata."""
    gesture: GestureType
    position_screen: np.ndarray      # (u, v) pixel coords of gesture centre
    position_normalized: np.ndarray  # (x, y, z) normalised coords
    confidence: float                # 0.0 – 1.0
    timestamp: float                 # time.time()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class GestureDetector:
    """
    Stateful gesture detector.

    Parameters
    ----------
    pinch_threshold : float
        Maximum normalised distance between two fingertips for a "pinch".
        Tuned for typical webcam distances; increase if gestures feel too
        tight, decrease if accidental triggers occur.
    cooldown : float
        Seconds to wait before recognising the *same* gesture again.
        Prevents rapid-fire placement / deletion while pinching.
    palm_spread : float
        Minimum normalised distance from each fingertip to the palm centre
        for a finger to be considered "extended".
    """

    def __init__(
        self,
        pinch_threshold: float = 0.055,
        cooldown: float = 0.45,
        palm_spread: float = 0.10,
    ):
        self.pinch_threshold = pinch_threshold
        self.cooldown = cooldown
        self.palm_spread = palm_spread

        self._last_place_time: float = 0.0
        self._last_delete_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, hand: HandLandmarks | None) -> GestureState:
        """
        Analyse a single hand and return the current gesture state.

        Priority order (first match wins):
          1. PINCH_DELETE  (thumb + middle, index extended)
          2. PINCH_PLACE   (thumb + index)
          3. OPEN_PALM     (≥ 4 fingers spread)
          4. NONE
        """
        now = time.time()

        if hand is None:
            return self._none_state(now)

        # --- distances ---------------------------------------------------
        d_thumb_index  = float(np.linalg.norm(hand.thumb_tip  - hand.index_tip))
        d_thumb_middle = float(np.linalg.norm(hand.thumb_tip  - hand.middle_tip))

        # 1. Delete pinch: thumb+middle close, index far (extended)
        if (
            d_thumb_middle < self.pinch_threshold
            and d_thumb_index > self.pinch_threshold * 1.8
            and now - self._last_delete_time > self.cooldown
        ):
            self._last_delete_time = now
            mid_screen = (hand.thumb_tip_screen + hand.middle_tip_screen) / 2.0
            mid_norm   = (hand.thumb_tip + hand.middle_tip) / 2.0
            return GestureState(
                gesture=GestureType.PINCH_DELETE,
                position_screen=mid_screen,
                position_normalized=mid_norm,
                confidence=max(0.0, 1.0 - d_thumb_middle / self.pinch_threshold),
                timestamp=now,
            )

        # 2. Place pinch: thumb+index close
        if (
            d_thumb_index < self.pinch_threshold
            and now - self._last_place_time > self.cooldown
        ):
            self._last_place_time = now
            mid_screen = (hand.thumb_tip_screen + hand.index_tip_screen) / 2.0
            mid_norm   = (hand.thumb_tip + hand.index_tip) / 2.0
            return GestureState(
                gesture=GestureType.PINCH_PLACE,
                position_screen=mid_screen,
                position_normalized=mid_norm,
                confidence=max(0.0, 1.0 - d_thumb_index / self.pinch_threshold),
                timestamp=now,
            )

        # 3. Open palm
        if self._is_open_palm(hand):
            return GestureState(
                gesture=GestureType.OPEN_PALM,
                position_screen=hand.palm_center_screen.copy(),
                position_normalized=hand.palm_center.copy(),
                confidence=0.9,
                timestamp=now,
            )

        return self._none_state(now, hand)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_open_palm(self, hand: HandLandmarks) -> bool:
        """
        Detect open palm by checking that at least 4 of 5 fingertips are
        extended (sufficiently far from the palm centre).
        """
        tip_indices = [4, 8, 12, 16, 20]   # thumb → pinky
        palm = hand.palm_center
        extended = sum(
            1
            for idx in tip_indices
            if np.linalg.norm(hand.all_landmarks[idx] - palm) > self.palm_spread
        )
        return extended >= 4

    @staticmethod
    def _none_state(ts: float, hand: HandLandmarks | None = None) -> GestureState:
        pos_s = hand.palm_center_screen.copy() if hand else np.zeros(2)
        pos_n = hand.palm_center.copy() if hand else np.zeros(3)
        return GestureState(
            gesture=GestureType.NONE,
            position_screen=pos_s,
            position_normalized=pos_n,
            confidence=0.0,
            timestamp=ts,
        )
