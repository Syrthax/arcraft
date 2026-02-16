"""
Rendering Module
================
GPU-accelerated 3D rendering with **ModernGL** (OpenGL 3.3 core).

Why ModernGL?
~~~~~~~~~~~~~
* **PyOpenGL** – C-style verbosity, manual buffer management, error-prone.
* **pyglet**   – game-oriented, less control over custom render pipelines.
* **ModernGL** – clean Pythonic API, direct shader access, first-class
  NumPy integration, instanced drawing, minimal boilerplate.  Ideal for
  a custom AR compositing pipeline.

Architecture
~~~~~~~~~~~~
1. **Background pass** – fullscreen quad textured with the webcam frame
   (depth-write disabled so cubes render on top).
2. **Grid pass** – translucent reference grid on the Y = 0 plane.
3. **Cube pass** – instanced rendering of unit cubes with per-instance
   position + colour.  Diffuse + ambient lighting.
4. **Ghost pass** – single translucent cube showing placement preview.
"""

from __future__ import annotations

import numpy as np
import moderngl
from typing import Optional, Tuple

import cv2  # used only for BGR→RGB conversion in update_background

# ======================================================================
# GLSL shader sources
# ======================================================================

# ── Background (fullscreen textured quad) ─────────────────────────────

_BG_VERT = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_BG_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D tex;
void main() {
    frag = texture(tex, v_uv);
}
"""

# ── Cubes (instanced, lit) ────────────────────────────────────────────

_CUBE_VERT = """
#version 330 core

in vec3 in_pos;
in vec3 in_normal;

// per-instance
in vec3 i_offset;
in vec3 i_color;

uniform mat4 u_proj;
uniform mat4 u_view;

out vec3 v_normal;
out vec3 v_color;
out vec3 v_world;

void main() {
    vec3 wp = in_pos + i_offset;
    gl_Position = u_proj * u_view * vec4(wp, 1.0);
    v_normal = in_normal;
    v_color  = i_color;
    v_world  = wp;
}
"""

_CUBE_FRAG = """
#version 330 core

in vec3 v_normal;
in vec3 v_color;
in vec3 v_world;

out vec4 frag;

uniform vec3 u_light_dir;
uniform vec3 u_light_col;
uniform vec3 u_ambient;

void main() {
    float diff = max(dot(normalize(v_normal), normalize(u_light_dir)), 0.0);
    vec3 lit = (u_ambient + diff * u_light_col) * v_color;
    frag = vec4(lit, 1.0);
}
"""

# ── Ghost cube (single translucent placement preview) ─────────────────

_GHOST_VERT = """
#version 330 core
in vec3 in_pos;
uniform mat4 u_proj;
uniform mat4 u_view;
uniform vec3 u_offset;
void main() {
    gl_Position = u_proj * u_view * vec4(in_pos + u_offset, 1.0);
}
"""

_GHOST_FRAG = """
#version 330 core
out vec4 frag;
uniform vec4 u_color;
void main() {
    frag = u_color;
}
"""

# ── Grid lines ────────────────────────────────────────────────────────

_GRID_VERT = """
#version 330 core
in vec3 in_pos;
uniform mat4 u_proj;
uniform mat4 u_view;
void main() {
    gl_Position = u_proj * u_view * vec4(in_pos, 1.0);
}
"""

_GRID_FRAG = """
#version 330 core
out vec4 frag;
uniform vec4 u_color;
void main() {
    frag = u_color;
}
"""

# ======================================================================
# Geometry helpers
# ======================================================================

def _unit_cube_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Unit cube spanning (0,0,0)→(1,1,1).
    Returns (vertices, indices).
    Vertex layout: 3f position, 3f normal   (interleaved, 24 B/vertex).
    """
    V = np.array([
        # front  z=1
        0,0,1, 0,0,1,  1,0,1, 0,0,1,  1,1,1, 0,0,1,  0,1,1, 0,0,1,
        # back   z=0
        1,0,0, 0,0,-1, 0,0,0, 0,0,-1, 0,1,0, 0,0,-1, 1,1,0, 0,0,-1,
        # top    y=1
        0,1,1, 0,1,0,  1,1,1, 0,1,0,  1,1,0, 0,1,0,  0,1,0, 0,1,0,
        # bottom y=0
        0,0,0, 0,-1,0, 1,0,0, 0,-1,0, 1,0,1, 0,-1,0, 0,0,1, 0,-1,0,
        # right  x=1
        1,0,1, 1,0,0,  1,0,0, 1,0,0,  1,1,0, 1,0,0,  1,1,1, 1,0,0,
        # left   x=0
        0,0,0,-1,0,0,  0,0,1,-1,0,0,  0,1,1,-1,0,0,  0,1,0,-1,0,0,
    ], dtype=np.float32)

    I = np.array([
         0, 1, 2,  0, 2, 3,
         4, 5, 6,  4, 6, 7,
         8, 9,10,  8,10,11,
        12,13,14, 12,14,15,
        16,17,18, 16,18,19,
        20,21,22, 20,22,23,
    ], dtype=np.int32)
    return V, I


def _grid_lines(half: int = 10) -> np.ndarray:
    """Line vertices for a reference grid on Y=0."""
    segs = []
    for i in range(-half, half + 1):
        segs += [i, 0, -half,  i, 0, half]   # lines ∥ Z
        segs += [-half, 0, i,  half, 0, i]   # lines ∥ X
    return np.array(segs, dtype=np.float32)


# ======================================================================
# Matrix helpers (pure NumPy, no deps)
# ======================================================================

def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)

    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -s.dot(eye)
    M[1, 3] = -u.dot(eye)
    M[2, 3] = f.dot(eye)
    return M


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = 2.0 * far * near / (near - far)
    M[3, 2] = -1.0
    return M


def _mat_bytes(m: np.ndarray) -> bytes:
    """Pack row-major NumPy matrix → column-major bytes for GLSL."""
    return m.astype(np.float32).tobytes(order='F')


# ======================================================================
# Camera
# ======================================================================

class Camera:
    """Orbital perspective camera."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up     = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Orbital state
        self.yaw      = -0.4   # radians, around Y
        self.pitch     = 0.55  # radians, elevation
        self.distance  = 14.0

        self.fov  = 55.0
        self.near = 0.1
        self.far  = 200.0

        self.position = np.zeros(3, dtype=np.float32)
        self._sync()

    # ---- controls ---------------------------------------------------

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw   += dx * 3.0
        self.pitch  = float(np.clip(self.pitch + dy * 2.0, 0.05, np.pi / 2 - 0.05))
        self._sync()

    def zoom(self, delta: float) -> None:
        self.distance = float(np.clip(self.distance + delta, 3.0, 60.0))
        self._sync()

    # ---- matrices ---------------------------------------------------

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.position, self.target, self.up)

    def proj_matrix(self) -> np.ndarray:
        return _perspective(self.fov, self.width / self.height, self.near, self.far)

    # ---- screen → world mapping -------------------------------------

    def screen_to_grid(
        self, sx: float, sy: float, grid_y: float = 0.0,
    ) -> Optional[Tuple[int, int, int]]:
        """
        Cast a ray from screen pixel *(sx, sy)* through the camera and
        intersect with the horizontal plane at ``y = grid_y``.

        Returns snapped ``(gx, gy, gz)`` or *None* if the ray misses.
        """
        # NDC
        nx = 2.0 * sx / self.width  - 1.0
        ny = 1.0 - 2.0 * sy / self.height

        vp = self.proj_matrix() @ self.view_matrix()
        inv = np.linalg.inv(vp)

        def _unproj(z: float) -> np.ndarray:
            v = inv @ np.array([nx, ny, z, 1.0], dtype=np.float32)
            return v[:3] / v[3]

        near_pt = _unproj(-1.0)
        far_pt  = _unproj( 1.0)
        d = far_pt - near_pt

        if abs(d[1]) < 1e-7:
            return None
        t = (grid_y - near_pt[1]) / d[1]
        if t < 0:
            return None
        hit = near_pt + t * d
        return (int(np.floor(hit[0])), int(grid_y), int(np.floor(hit[2])))

    # ---- internal ---------------------------------------------------

    def _sync(self) -> None:
        cp, cy = np.cos(self.pitch), np.cos(self.yaw)
        sp, sy = np.sin(self.pitch), np.sin(self.yaw)
        self.position[0] = self.target[0] + self.distance * cp * sy
        self.position[1] = self.target[1] + self.distance * sp
        self.position[2] = self.target[2] + self.distance * cp * cy


# ======================================================================
# Renderer
# ======================================================================

class Renderer:
    """
    Manages all OpenGL state and draws each frame.

    Typical per-frame usage::

        renderer.update_background(bgr_frame)
        renderer.update_cubes(world.get_render_data())
        renderer.set_ghost(grid_pos)        # or clear_ghost()
        renderer.render()
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self.ctx = ctx
        self.width = width
        self.height = height
        self.camera = Camera(width, height)

        self.ctx.enable(moderngl.DEPTH_TEST)

        # ---- background pass ----------------------------------------
        self._bg_prog = ctx.program(vertex_shader=_BG_VERT, fragment_shader=_BG_FRAG)
        quad = np.array([
            # pos(x,y)  uv(u,v)
            -1, -1,  0, 1,
             1, -1,  1, 1,
            -1,  1,  0, 0,
             1,  1,  1, 0,
        ], dtype=np.float32)
        bg_vbo = ctx.buffer(quad.tobytes())
        self._bg_vao = ctx.vertex_array(
            self._bg_prog, [(bg_vbo, '2f 2f', 'in_pos', 'in_uv')],
        )
        # Placeholder texture – real size set on first frame
        self._bg_tex: Optional[moderngl.Texture] = None
        self._tex_size: Tuple[int, int] = (0, 0)

        # ---- cube pass (instanced) ----------------------------------
        self._cube_prog = ctx.program(vertex_shader=_CUBE_VERT, fragment_shader=_CUBE_FRAG)
        self._cube_prog['u_light_dir'].value = (0.4, 0.9, 0.3)
        self._cube_prog['u_light_col'].value = (0.75, 0.75, 0.72)
        self._cube_prog['u_ambient'].value   = (0.35, 0.35, 0.38)

        verts, idxs = _unit_cube_data()
        self._cube_vbo = ctx.buffer(verts.tobytes())
        self._cube_ibo = ctx.buffer(idxs.tobytes())
        self._inst_vbo = ctx.buffer(reserve=6 * 4 * 2048)  # room for 2 048 cubes
        self._inst_count: int = 0
        self._cube_vao: Optional[moderngl.VertexArray] = None

        # ---- ghost cube pass -----------------------------------------
        self._ghost_prog = ctx.program(vertex_shader=_GHOST_VERT, fragment_shader=_GHOST_FRAG)
        ghost_vbo = ctx.buffer(verts.tobytes())
        self._ghost_vao = ctx.vertex_array(
            self._ghost_prog,
            [(ghost_vbo, '3f 3x4', 'in_pos')],   # skip normals
            index_buffer=ctx.buffer(idxs.tobytes()),
        )
        self._ghost_pos: Optional[Tuple[int, int, int]] = None

        # ---- grid pass -----------------------------------------------
        self._grid_prog = ctx.program(vertex_shader=_GRID_VERT, fragment_shader=_GRID_FRAG)
        gv = _grid_lines(10)
        gvbo = ctx.buffer(gv.tobytes())
        self._grid_vao = ctx.vertex_array(
            self._grid_prog, [(gvbo, '3f', 'in_pos')],
        )
        self._grid_vcnt = len(gv) // 3
        self._grid_prog['u_color'].value = (0.45, 0.45, 0.45, 0.35)

    # ------------------------------------------------------------------
    # Per-frame data updates
    # ------------------------------------------------------------------

    def update_background(self, frame_bgr: np.ndarray) -> None:
        """Upload a BGR webcam frame as the background texture."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # (Re-)create texture when resolution changes
        if (w, h) != self._tex_size:
            if self._bg_tex is not None:
                self._bg_tex.release()
            self._bg_tex = self.ctx.texture((w, h), 3)
            self._bg_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._tex_size = (w, h)

        self._bg_tex.write(rgb.tobytes())

    def update_cubes(self, data: np.ndarray) -> None:
        """
        Upload instance data from ``World.get_render_data()``
        (Nx6 float32: x y z r g b).
        """
        self._inst_count = len(data)
        if self._inst_count == 0:
            return

        raw = data.tobytes()
        if len(raw) > self._inst_vbo.size:
            self._inst_vbo = self.ctx.buffer(raw)
        else:
            self._inst_vbo.write(raw)

        self._cube_vao = self.ctx.vertex_array(
            self._cube_prog,
            [
                (self._cube_vbo, '3f 3f', 'in_pos', 'in_normal'),
                (self._inst_vbo, '3f 3f /i', 'i_offset', 'i_color'),
            ],
            index_buffer=self._cube_ibo,
        )

    def set_ghost(self, pos: Tuple[int, int, int]) -> None:
        self._ghost_pos = pos

    def clear_ghost(self) -> None:
        self._ghost_pos = None

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Draw one complete frame: background → grid → cubes → ghost."""
        self.ctx.clear(0.08, 0.08, 0.12, 1.0)

        view = self.camera.view_matrix()
        proj = self.camera.proj_matrix()
        vb = _mat_bytes(view)
        pb = _mat_bytes(proj)

        # 1. Background (no depth write)
        # Depth test is disabled → background won't write to the depth buffer.
        # The initial clear() already set depth to 1.0 everywhere, so cubes
        # rendered later will always be "in front" of the background.
        self.ctx.disable(moderngl.DEPTH_TEST)
        if self._bg_tex is not None:
            self._bg_tex.use(0)
            self._bg_prog['tex'].value = 0
            self._bg_vao.render(moderngl.TRIANGLE_STRIP)

        # 2. Re-enable depth testing for 3-D content
        #    (depth buffer is still 1.0 from the initial clear—no second
        #     clear needed; a second clear() would also wipe the colour buffer)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # 3. Grid
        self._grid_prog['u_proj'].write(pb)
        self._grid_prog['u_view'].write(vb)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._grid_vao.render(moderngl.LINES)

        # 4. Cubes
        if self._inst_count > 0 and self._cube_vao is not None:
            self._cube_prog['u_proj'].write(pb)
            self._cube_prog['u_view'].write(vb)
            self._cube_vao.render(instances=self._inst_count)

        # 5. Ghost cube (translucent)
        if self._ghost_pos is not None:
            self._ghost_prog['u_proj'].write(pb)
            self._ghost_prog['u_view'].write(vb)
            self._ghost_prog['u_offset'].value = tuple(float(v) for v in self._ghost_pos)
            self._ghost_prog['u_color'].value = (0.3, 1.0, 0.3, 0.35)
            self._ghost_vao.render()

        self.ctx.disable(moderngl.BLEND)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def resize(self, w: int, h: int) -> None:
        if w < 1 or h < 1:
            return
        self.width, self.height = w, h
        self.camera.width, self.camera.height = w, h
        self.ctx.viewport = (0, 0, w, h)

    def cleanup(self) -> None:
        """Explicit cleanup (context destruction handled externally)."""
        pass
