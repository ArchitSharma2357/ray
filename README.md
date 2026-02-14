# Raycism Engine

`Raycism Engine` is a standalone C++ path tracer focused on fast CPU/GPU iteration and studio-style preview workflows.

## Portable Setup (Windows + Linux)

Install dependencies and build binaries with one command:

```bash
python scripts/orion_portable.py install
```

Run the web app with one command:

```bash
python scripts/orion_portable.py run
```

Install: `python scripts/orion_portable.py install`  
Run: `python scripts/orion_portable.py run`

Optional native wrappers:

- Windows (PowerShell/CMD): `.\install.bat` and `.\run.bat`
- Linux/macOS: `./scripts/install.sh` and `./scripts/run.sh`

### Windows Native Quickstart (No WSL Required)

From the project directory:

```powershell
cd <repo-root>\orion-raytracer
.\install.bat
.\run.bat
```

Then open `http://127.0.0.1:8092`.

If your Python environment blocks `pip` writes (managed/system Python), run:

```bash
python scripts/orion_portable.py install --skip-pip
```

and pre-install required Python packages separately.

## Features

- Physically based Monte Carlo path tracing
- Lambertian, metal, dielectric, and emissive materials
- BVH acceleration structure for faster intersection tests
- Optional CUDA GPU backend (`--backend gpu`) with CPU fallback
- `orion_studio` interactive frontend with progressive live preview
- Clean studio preview frontend (no decorative overlay)
- Thin-lens camera (depth of field)
- Multi-threaded tileless scanline rendering
- ACES-like tone mapping and gamma correction
- Simple command line controls

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
./build/orion_raytracer
```

Useful options:

```bash
./build/orion_raytracer \
  --width 1280 --height 720 --samples 400 --depth 18 \
  --threads 8 --seed 7 --backend auto --demo cube_room --output out/cube_room.ppm
```

For lower logging overhead (useful in live loops):

```bash
./build/orion_raytracer --quiet --demo cube_room
```

Output is a binary PPM image.

Available demo presets:

- `cube_room` (default): single cube in a lit room
- `showcase`: dense material field and hero spheres
- `showcase_closeup`: closer framing for material/reflection checks (GPU-ready)
- `showcase_wide`: wide-angle framing for composition checks (GPU-ready)
- `showcase_topdown`: top-down analytical framing (GPU-ready)
- `hero_spheres`: simplified material test scene
- `cube_ring`: center cube with surrounding cube ring

## Live Frontend

Launch the interactive studio frontend:

```bash
./build/orion_studio --backend auto --width 1280 --height 720
```

For smoother mouse look, tune:

```bash
./build/orion_studio \
  --backend auto --mouse-sensitivity 0.12 --mouse-smoothing 0.65
```

Controls:

- `RMB` drag: look around
- `W A S D` + `Q/E`: move camera
- `Shift`: faster movement
- `Z/X`: field of view down/up
- `R/F`: focus distance up/down
- `C/V`: aperture up/down
- `Esc`: quit

The preview updates continuously at low sample count while moving, then refines progressively when camera motion stops.

## Web Frontend (Raycism-Specific)

Raycism Engine includes a dedicated modern web frontend (separate from the main SORT frontend).

Run it with:

```bash
python scripts/orion_portable.py run
```

Then open:

```text
http://127.0.0.1:8092
```

What it controls:

- `orion_raytracer` final render mode
- `orion_studio` live mode
- shared controls for backend, resolution, depth/sample settings
- built-in demo preset selection (`cube_room`, `showcase`, `showcase_closeup`, `showcase_wide`, `showcase_topdown`, `hero_spheres`, `cube_ring`)
- live process logs, command preview, and output file browser
- adaptive near-live loop (fast preview passes + periodic full-quality refinement)
- one-click live quality profiles (`speed`, `balanced`, `ultra`)

Notes:

- Build binaries first if missing:
  - `cmake -S . -B build`
  - `cmake --build build -j`
- Browser image preview is shown for image types with native browser support.
  For `.ppm`/`.exr` outputs, use the output file links to open/download.

## Backends

- `--backend auto` (default): uses CUDA if compiled in, otherwise CPU.
- `--backend cpu`: force CPU renderer.
- `--backend gpu`: force CUDA renderer (fails if CUDA backend is not compiled).

To build GPU support, install NVIDIA CUDA toolkit first, then re-run CMake configure/build.
