# ARCraft — Augmented Reality Block Builder

A production-quality prototype of a **Minecraft-style Augmented Reality block-placement system** powered by hand tracking. Place, delete, and interact with 3D cubes using pinch and palm gestures through your webcam — no headset or game engine required.

<p align="center">
  <b>Pure Python &bull; Real 3D rendering &bull; Live webcam AR overlay &bull; Hand gesture control</b>
</p>

---

## Features

- **Real-time hand tracking** via MediaPipe Hand Landmarker (Tasks API)
- **Gesture recognition** — pinch to place, two-finger pinch to delete, open palm to orbit camera
- **True 3D cube rendering** with perspective projection, depth buffering, and diffuse lighting (not 2D rectangles)
- **Live webcam AR overlay** — cubes render on top of your camera feed
- **Infinite sparse voxel grid** — cubes stored in a dictionary-backed world with no fixed boundaries
- **Auto-stacking** — new cubes stack on top of existing ones
- **Ghost preview** — translucent cube shows where the next block will land
- **Save / Load** — persist your world to JSON and reload it later
- **Reference grid** — translucent ground grid for spatial orientation
- **Keyboard & scroll fallbacks** — full camera control even without hand gestures
- **30+ FPS** on modern hardware

---

## Tech Stack

| Layer | Library | Why |
|-------|---------|-----|
| Camera input | **OpenCV** | Industry standard for webcam capture and image processing |
| Hand tracking | **MediaPipe** (Tasks API) | Google's hand landmark model — fast, accurate, no GPU required |
| 3D rendering | **ModernGL** | Clean Pythonic OpenGL 3.3 wrapper with first-class NumPy support and instanced drawing — superior to PyOpenGL (too verbose) and pyglet (less pipeline control) |
| Window / input | **GLFW** | Lightweight cross-platform window + GL context creation |
| Math | **NumPy** | Fast matrix/vector operations for projection, view, and ray-casting |

---

## Project Structure

```
arcraft/
├── main.py              # Application entry point & game loop
├── hand_tracking.py     # MediaPipe hand landmark wrapper
├── gestures.py          # Gesture detection (pinch / palm / delete)
├── renderer.py          # ModernGL 3D rendering pipeline
├── world.py             # Sparse voxel world + JSON save/load
├── hand_landmarker.task # MediaPipe model file (downloaded during setup)
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- **Python 3.9+** (tested on 3.9 and 3.11)
- **macOS / Linux / Windows** (macOS tested; OpenGL 3.3 core profile required)
- A **webcam**

### 1. Clone the repository

```bash
git clone https://github.com/Syrthax/arcraft.git
cd arcraft
```

### 2. (Recommended) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the hand tracking model

The MediaPipe hand landmarker model (`hand_landmarker.task`) must be in the project root. If it's not already present:

```bash
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 5. Run

```bash
python3 main.py
```

A window titled **ARCraft** will open showing your webcam feed with a 3D grid and starter cubes overlaid on top.

---

## Gesture Guide

All gestures are performed with one hand in view of the webcam.

### Place Cube — Thumb + Index Pinch 🤏

Bring your **thumb tip** and **index finger tip** close together (distance < threshold).

- A cube is placed at the grid position your pinch is aimed at.
- Cubes auto-stack: if the target cell is occupied, the new cube is placed on top.
- A **translucent green ghost cube** shows where the block will land before you pinch.

### Delete Cube — Thumb + Middle Pinch ✌️

Bring your **thumb tip** and **middle finger tip** together while keeping your **index finger extended**.

- The topmost cube at the aimed grid column is removed.
- The extended index finger disambiguates this from the place gesture.

### Orbit Camera — Open Palm 🖐️

Spread all five fingers (open palm). Move your hand to **orbit the camera** around the scene.

- The camera rotates to follow your palm movement.
- Release (close hand or move out of view) to stop orbiting.

### Gesture Cooldown

Each place/delete gesture has a **0.45-second cooldown** to prevent accidental rapid-fire. Hold the pinch steady — the next trigger happens automatically after the cooldown.

---

## Keyboard & Mouse Controls

| Key | Action |
|-----|--------|
| `ESC` | Quit (auto-saves world) |
| `S` | Save world to `world_save.json` |
| `L` | Load world from `world_save.json` |
| `C` | Clear all cubes |
| `R` | Reset camera to default position |
| `←` / `→` | Orbit camera left / right |
| `↑` / `↓` | Zoom in / out |
| **Scroll wheel** | Zoom in / out |

---

## Save & Load

The world is automatically saved to `world_save.json` on exit and can be manually saved/loaded with `S` / `L` keys.

The save file is human-readable JSON:

```json
{
  "version": 1,
  "cube_count": 25,
  "color_index": 0,
  "cubes": [
    { "position": [0, 0, 0], "color": [0.36, 0.71, 0.36] },
    ...
  ]
}
```

---

## Architecture

### Rendering Pipeline (per frame)

1. **Clear** — colour buffer (dark) + depth buffer (1.0)
2. **Background pass** — fullscreen quad with webcam texture, depth test OFF (doesn't write depth)
3. **Grid pass** — translucent reference grid at Y = 0, blending ON
4. **Cube pass** — instanced rendering of all cubes with diffuse + ambient lighting
5. **Ghost pass** — single translucent cube at the aimed placement position

### Coordinate Flow

```
MediaPipe normalised (0-1)
  → screen pixels (webcam resolution)
    → scaled to window pixels
      → ray cast through camera (NDC → world)
        → intersect Y = 0 ground plane
          → snap to integer grid
```

### Gesture Detection Logic

```
thumb_index_dist  = ‖thumb_tip − index_tip‖
thumb_middle_dist = ‖thumb_tip − middle_tip‖

if thumb_middle < threshold AND thumb_index > 1.8× threshold → DELETE
elif thumb_index < threshold                                  → PLACE
elif ≥ 4 fingertips far from palm centre                      → OPEN_PALM
else                                                          → NONE
```

---

## Performance Notes

- MediaPipe runs with the **lightweight model** (`model_complexity=0` equivalent via the float16 `.task` file)
- Cube geometry uses **instanced rendering** — one draw call for all cubes
- World render data is **cached** and only rebuilt when cubes are added or removed (dirty flag)
- Webcam texture is uploaded directly to the GPU each frame (no CPU-side compositing)
- Target: **30+ FPS** on modern laptops

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: hand_landmarker.task` | Download the model — see Installation step 4 |
| Black window / no webcam | Check webcam permissions; try a different `cv2.VideoCapture(1)` index in `main.py` |
| `GLFW init failed` | Install GLFW system library: `brew install glfw` (macOS) or `sudo apt install libglfw3` (Linux) |
| Low FPS | Close other camera apps; reduce `CAM_W`/`CAM_H` in `main.py` |
| Cubes not appearing | Make sure you're pinching thumb + index firmly; adjust `pinch_threshold` in `gestures.py` |

---

## License

MIT

---

*Built with Python, OpenCV, MediaPipe, ModernGL, and GLFW.*
A block game using Augmented Reality, made coz 50% of the syllabus for tomorrow's exam is done and  I need some rest
