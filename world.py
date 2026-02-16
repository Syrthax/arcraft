"""
World System Module
===================
Manages the persistent voxel world: an **infinite sparse grid** backed by a
Python dictionary keyed on integer ``(x, y, z)`` tuples.

Features
--------
* Add / remove cubes at arbitrary integer grid positions.
* Automatic Minecraft-style colour palette cycling.
* Render-data cache – only rebuilt when cubes change (dirty flag).
* JSON serialize / deserialize for save & load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (Minecraft-inspired)
# ---------------------------------------------------------------------------

CUBE_COLORS: List[Tuple[float, float, float]] = [
    (0.36, 0.71, 0.36),   # Grass green
    (0.55, 0.37, 0.24),   # Dirt brown
    (0.62, 0.62, 0.62),   # Stone grey
    (0.82, 0.71, 0.45),   # Sand
    (0.30, 0.50, 0.80),   # Water blue
    (0.45, 0.30, 0.18),   # Dark wood
    (0.88, 0.88, 0.88),   # Snow white
    (0.90, 0.30, 0.20),   # Brick red
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CubeData:
    """Single cube stored in the world."""
    position: Tuple[int, int, int]
    color: Tuple[float, float, float]   # RGB each 0.0–1.0

    def to_dict(self) -> dict:
        return {"position": list(self.position), "color": list(self.color)}

    @classmethod
    def from_dict(cls, data: dict) -> CubeData:
        return cls(
            position=tuple(data["position"]),
            color=tuple(data["color"]),
        )


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class World:
    """
    Sparse voxel world.

    Storage
    ~~~~~~~
    ``_cubes`` is ``Dict[Tuple[int,int,int], CubeData]``.
    Only occupied cells consume memory → effectively infinite grid.

    Dirty flag
    ~~~~~~~~~~
    ``_dirty`` is set whenever cubes are added / removed.  The renderer
    checks ``is_dirty`` and calls ``get_render_data()`` to get a fresh
    NumPy array.  The array is cached until the next mutation.
    """

    def __init__(self) -> None:
        self._cubes: Dict[Tuple[int, int, int], CubeData] = {}
        self._color_idx: int = 0
        self._dirty: bool = True
        self._render_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_cube(
        self,
        x: int, y: int, z: int,
        color: Optional[Tuple[float, float, float]] = None,
    ) -> bool:
        """Place a cube.  Returns *False* if the cell is already occupied."""
        key = (int(x), int(y), int(z))
        if key in self._cubes:
            return False

        if color is None:
            color = CUBE_COLORS[self._color_idx % len(CUBE_COLORS)]
            self._color_idx += 1

        self._cubes[key] = CubeData(position=key, color=color)
        self._dirty = True
        return True

    def remove_cube(self, x: int, y: int, z: int) -> bool:
        """Remove a cube.  Returns *False* if the cell was empty."""
        key = (int(x), int(y), int(z))
        if key not in self._cubes:
            return False
        del self._cubes[key]
        self._dirty = True
        return True

    def clear(self) -> None:
        """Delete every cube in the world."""
        self._cubes.clear()
        self._color_idx = 0
        self._dirty = True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_cube(self, x: int, y: int, z: int) -> bool:
        return (x, y, z) in self._cubes

    def get_cube(self, x: int, y: int, z: int) -> Optional[CubeData]:
        return self._cubes.get((x, y, z))

    def stack_height(self, x: int, z: int, max_h: int = 30) -> int:
        """Return the lowest unoccupied Y at column (x, z)."""
        for y in range(max_h):
            if not self.has_cube(x, y, z):
                return y
        return max_h

    def find_nearest_cube(
        self, x: float, y: float, z: float, max_dist: float = 2.0,
    ) -> Optional[Tuple[int, int, int]]:
        """Return grid key of nearest cube within *max_dist*, or None."""
        best, best_d = None, float("inf")
        pt = np.array([x, y, z])
        for key in self._cubes:
            d = float(np.linalg.norm(pt - np.array(key, dtype=float)))
            if d < best_d and d <= max_dist:
                best, best_d = key, d
        return best

    def get_all_cubes(self) -> List[CubeData]:
        return list(self._cubes.values())

    # ------------------------------------------------------------------
    # Render data
    # ------------------------------------------------------------------

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def cube_count(self) -> int:
        return len(self._cubes)

    def get_render_data(self) -> np.ndarray:
        """
        Return an ``(N, 6)`` float32 array: ``[x, y, z, r, g, b]`` per cube.
        Result is cached until the world mutates.
        """
        if not self._dirty and self._render_cache is not None:
            return self._render_cache

        if not self._cubes:
            self._render_cache = np.empty((0, 6), dtype=np.float32)
        else:
            rows = [
                [*c.position, *c.color] for c in self._cubes.values()
            ]
            self._render_cache = np.array(rows, dtype=np.float32)

        self._dirty = False
        return self._render_cache

    # ------------------------------------------------------------------
    # Persistence  (JSON)
    # ------------------------------------------------------------------

    def save(self, filepath: str = "world_save.json") -> None:
        """Serialise the world to a JSON file."""
        payload = {
            "version": 1,
            "cube_count": len(self._cubes),
            "color_index": self._color_idx,
            "cubes": [c.to_dict() for c in self._cubes.values()],
        }
        Path(filepath).write_text(json.dumps(payload, indent=2))
        print(f"[World] Saved {len(self._cubes)} cube(s) → {filepath}")

    def load(self, filepath: str = "world_save.json") -> bool:
        """
        Load world from JSON.  Returns *True* on success.
        """
        path = Path(filepath)
        if not path.exists():
            print(f"[World] No save file found at {filepath}")
            return False

        try:
            payload = json.loads(path.read_text())
            self._cubes.clear()
            self._color_idx = payload.get("color_index", 0)
            for entry in payload["cubes"]:
                cube = CubeData.from_dict(entry)
                self._cubes[cube.position] = cube
            self._dirty = True
            print(f"[World] Loaded {len(self._cubes)} cube(s) ← {filepath}")
            return True
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[World] Failed to load {filepath}: {exc}")
            return False
