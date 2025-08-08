# 3D Tonnetz Visualizer

Interactive 3D Tonnetz lattice for pitch classes and triads, using Plotly.

## Install

Use your existing virtualenv (recommended):

```sh
pip install -e .
```

Or just install deps to run a one-off script:

```sh
pip install -r requirements.txt
```

## CLI

Generate an HTML file (default: `tonnetz.html`):

```sh
tonnetz3d --radius 2 --root C --quality maj --out tonnetz.html
```

Open an interactive window (browser):

```sh
tonnetz3d --show --root A --quality min
```

Options:
- `--radius` lattice radius (>=1)
- `--root` root note, e.g., C, F#, Eb
- `--quality` triad quality: maj|min
- `--no-sharps` prefer flats in labels
- `--show` open in a browser instead of writing HTML
- `--out` output HTML path

## Python API

```python
from tonnetz3d import Tonnetz3D

viz = Tonnetz3D(radius=2)
fig = viz.figure(highlight=("C", "maj"))
fig.show()
```

## What is the Tonnetz?

The Tonnetz is a lattice that organizes pitch classes so that moves along axes correspond to consonant intervals (perfect fifths, major thirds, minor thirds). Triads appear as small triangles on the lattice.
