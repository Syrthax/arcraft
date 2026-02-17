#!/usr/bin/env python3
"""
ARCraft – Minecraft-style Augmented Reality Block Builder
=========================================================

Main application entry-point.

Controls
--------
**Hand gestures (webcam)**
  * Thumb + index pinch  → place cube
  * Thumb + middle pinch → delete cube
  * Open palm            → orbit camera

**Keyboard**
  * ESC       – quit
  * S         – save world
  * L         – load world
  * C         – clear world
  * R         – reset camera
  * ↑ / ↓     – zoom in / out
  * ← / →     – rotate camera
  * W / A / S / D (held) – also orbit camera

**Mouse**
  * Scroll    – zoom
"""

from __future__ import annotations

import sys
import time

import cv2
import glfw
import moderngl
import numpy as np

from gestures import GestureDetector, GestureType
from hand_tracking import HandTracker
from renderer import Renderer
from world import World


# ======================================================================
# Application
# ======================================================================

class Application:
    """Top-level application – owns the window, GL context, and game loop."""

    WIN_W = 960
    WIN_H = 720
    CAM_W = 640
    CAM_H = 480

    def __init__(self) -> None:
        # ---- GLFW window + GL context --------------------------------
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)   # required on macOS
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, False)

        self._win = glfw.create_window(self.WIN_W, self.WIN_H,
                                       "ARCraft", None, None)
        if not self._win:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._win)
        glfw.swap_interval(1)   # vsync

        self._ctx = moderngl.create_context()

        # ---- sub-systems ---------------------------------------------
        self._world    = World()
        self._renderer = Renderer(self._ctx, self.WIN_W, self.WIN_H)
        self._tracker  = HandTracker(max_hands=2)
        self._gestures = GestureDetector()

        # ---- webcam ---------------------------------------------------
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        if not self._cap.isOpened():
            print("[ARCraft] WARNING: webcam not available – running without camera")

        # ---- interaction state ----------------------------------------
        self._prev_palm: np.ndarray | None = None
        self._prev_palm_distance: float | None = None  # for two-hand zoom
        self._key_cooldown: dict[int, float] = {}

        # ---- FPS bookkeeping -----------------------------------------
        self._t0 = time.perf_counter()
        self._frames = 0
        self._fps = 0.0

        # ---- GLFW callbacks ------------------------------------------
        glfw.set_framebuffer_size_callback(self._win, self._cb_resize)
        glfw.set_scroll_callback(self._win, self._cb_scroll)

        # ---- initial world -------------------------------------------
        if not self._world.load():
            self._place_platform()

        self._renderer.update_cubes(self._world.get_render_data())

    # ------------------------------------------------------------------
    # Starter content
    # ------------------------------------------------------------------

    def _place_platform(self) -> None:
        """Lay a 5×5 grass platform at Y=0."""
        for x in range(-2, 3):
            for z in range(-2, 3):
                self._world.add_cube(x, 0, z, color=(0.36, 0.71, 0.36))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("[ARCraft] Running.  ESC to quit  |  S save  |  L load  |  C clear")

        while not glfw.window_should_close(self._win):
            self._tick()
            self._renderer.render()
            glfw.swap_buffers(self._win)
            glfw.poll_events()
            self._handle_keys()
            self._update_fps()

        self._shutdown()

    # ------------------------------------------------------------------
    # Per-frame logic
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        # 1. Grab webcam frame
        ok, frame = self._cap.read()
        if not ok:
            # No camera – render cubes only (black background)
            black = np.zeros((self.CAM_H, self.CAM_W, 3), dtype=np.uint8)
            self._renderer.update_background(black)
            self._renderer.clear_ghost()
            return

        frame = cv2.flip(frame, 1)   # mirror

        # 2. Hand tracking
        hands = self._tracker.process_frame(frame)

        # 3. Gesture + world update
        if hands:
            # Check for two-hand zoom gesture first
            if len(hands) == 2:
                self._process_two_hand_zoom(hands)
                label = "ZOOM"
            else:
                self._prev_palm_distance = None  # reset zoom state
                hand = hands[0]
                gs = self._gestures.detect(hand)
                self._process_gesture(gs, hand)
                label = gs.gesture.name

            # Draw landmarks on the webcam feed for visual feedback
            self._tracker.draw_landmarks(frame, hands)

            # Draw gesture label
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            self._prev_palm = None
            self._prev_palm_distance = None
            self._renderer.clear_ghost()

        # 4. Upload frame as background texture
        self._renderer.update_background(frame)

        # 5. Push cube render data if world changed
        if self._world.is_dirty:
            self._renderer.update_cubes(self._world.get_render_data())

    def _process_gesture(self, gs, hand) -> None:
        cam = self._renderer.camera

        # We need to map hand screen coords (webcam pixels) → window pixels
        wx_scale = self._renderer.width  / self._tracker.frame_size[0]
        wy_scale = self._renderer.height / self._tracker.frame_size[1]

        sx = gs.position_screen[0] * wx_scale
        sy = gs.position_screen[1] * wy_scale

        if gs.gesture == GestureType.PINCH_PLACE:
            grid = cam.screen_to_grid(sx, sy, grid_y=0)
            if grid:
                gx, _, gz = grid
                gy = self._world.stack_height(gx, gz)
                if self._world.add_cube(gx, gy, gz):
                    print(f"  ✚ Placed cube ({gx},{gy},{gz})")
            self._renderer.clear_ghost()

        elif gs.gesture == GestureType.PINCH_DELETE:
            grid = cam.screen_to_grid(sx, sy, grid_y=0)
            if grid:
                gx, _, gz = grid
                # delete topmost cube in the column
                for y in range(29, -1, -1):
                    if self._world.remove_cube(gx, y, gz):
                        print(f"  ✖ Deleted cube ({gx},{y},{gz})")
                        break
            self._renderer.clear_ghost()

        elif gs.gesture == GestureType.OPEN_PALM:
            palm = hand.palm_center_screen * np.array([wx_scale, wy_scale])
            if self._prev_palm is not None:
                dx = (palm[0] - self._prev_palm[0]) / self._renderer.width
                dy = (palm[1] - self._prev_palm[1]) / self._renderer.height
                cam.orbit(dx, dy)
            self._prev_palm = palm.copy()
            self._renderer.clear_ghost()

        else:
            self._prev_palm = None
            # Show ghost preview at aimed grid cell
            grid = cam.screen_to_grid(sx, sy, grid_y=0)
            if grid:
                gx, _, gz = grid
                gy = self._world.stack_height(gx, gz)
                self._renderer.set_ghost((gx, gy, gz))
            else:
                self._renderer.clear_ghost()

    def _process_two_hand_zoom(self, hands) -> None:
        """
        Process two-hand zoom gesture.
        Palms moving closer → zoom out
        Palms moving apart → zoom in
        """
        cam = self._renderer.camera

        # Get palm centers for both hands (in screen pixels)
        wx_scale = self._renderer.width / self._tracker.frame_size[0]
        wy_scale = self._renderer.height / self._tracker.frame_size[1]

        palm0 = hands[0].palm_center_screen * np.array([wx_scale, wy_scale])
        palm1 = hands[1].palm_center_screen * np.array([wx_scale, wy_scale])

        # Calculate current distance between palms
        current_distance = float(np.linalg.norm(palm0 - palm1))

        if self._prev_palm_distance is not None:
            # Calculate distance change
            delta = current_distance - self._prev_palm_distance
            # Scale the zoom effect (negative delta = zoom out, positive = zoom in)
            # Normalize by window width for consistent behavior
            zoom_factor = -delta / self._renderer.width * 30.0
            cam.zoom(zoom_factor)

        self._prev_palm_distance = current_distance
        self._renderer.clear_ghost()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _handle_keys(self) -> None:
        now = time.time()
        cam = self._renderer.camera

        def _pressed(key: int, cooldown: float = 0.3) -> bool:
            if glfw.get_key(self._win, key) != glfw.PRESS:
                return False
            if now - self._key_cooldown.get(key, 0) < cooldown:
                return False
            self._key_cooldown[key] = now
            return True

        if _pressed(glfw.KEY_ESCAPE, 0):
            glfw.set_window_should_close(self._win, True)
        if _pressed(glfw.KEY_S):
            self._world.save()
        if _pressed(glfw.KEY_L):
            self._world.load()
            self._renderer.update_cubes(self._world.get_render_data())
        if _pressed(glfw.KEY_C):
            self._world.clear()
            self._renderer.update_cubes(self._world.get_render_data())
            print("  World cleared")
        if _pressed(glfw.KEY_R):
            cam.yaw, cam.pitch, cam.distance = -0.4, 0.55, 14.0
            cam._sync()
            print("  Camera reset")

        # Continuous camera controls (no cooldown)
        step = 0.02
        if glfw.get_key(self._win, glfw.KEY_LEFT) == glfw.PRESS:
            cam.orbit(-step, 0)
        if glfw.get_key(self._win, glfw.KEY_RIGHT) == glfw.PRESS:
            cam.orbit(step, 0)
        if glfw.get_key(self._win, glfw.KEY_UP) == glfw.PRESS:
            cam.zoom(-0.3)
        if glfw.get_key(self._win, glfw.KEY_DOWN) == glfw.PRESS:
            cam.zoom(0.3)

    # ------------------------------------------------------------------
    # GLFW callbacks
    # ------------------------------------------------------------------

    def _cb_resize(self, _win, w: int, h: int) -> None:
        if w > 0 and h > 0:
            self._renderer.resize(w, h)

    def _cb_scroll(self, _win, _xoff: float, yoff: float) -> None:
        self._renderer.camera.zoom(-yoff)

    # ------------------------------------------------------------------
    # FPS
    # ------------------------------------------------------------------

    def _update_fps(self) -> None:
        self._frames += 1
        elapsed = time.perf_counter() - self._t0
        if elapsed >= 1.0:
            self._fps = self._frames / elapsed
            self._frames = 0
            self._t0 = time.perf_counter()
            glfw.set_window_title(
                self._win,
                f"ARCraft  |  {self._fps:.0f} FPS  |  {self._world.cube_count} cubes",
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        self._world.save()
        self._tracker.release()
        self._cap.release()
        self._renderer.cleanup()
        glfw.terminate()
        print("[ARCraft] Shutdown complete.  World auto-saved.")


# ======================================================================
# Entry point
# ======================================================================

def main() -> None:
    try:
        app = Application()
        app.run()
    except Exception as exc:
        print(f"\n[ARCraft] Fatal error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
