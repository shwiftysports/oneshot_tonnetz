from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

# ----------------------
# Note name helpers
# ----------------------

NOTE_TO_PC: Dict[str, int] = {
    # naturals
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
    # sharps
    "C#": 1,
    "D#": 3,
    "F#": 6,
    "G#": 8,
    "A#": 10,
    # flats
    "Db": 1,
    "Eb": 3,
    "Gb": 6,
    "Ab": 8,
    "Bb": 10,
}

PC_TO_NAME_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_TO_NAME_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def note_to_pc(note: str) -> int:
    s = note.strip().replace("♯", "#").replace("♭", "b")
    s = s.upper()
    # keep accidental case
    s = s[0] + s[1:] if len(s) > 1 else s
    if s in NOTE_TO_PC:
        return NOTE_TO_PC[s]
    raise ValueError(f"Unknown note name: {note}")


def pc_to_name(pc: int, prefer_sharps: bool = True) -> str:
    pc = pc % 12
    return (PC_TO_NAME_SHARP if prefer_sharps else PC_TO_NAME_FLAT)[pc]


# ----------------------
# Tonnetz core
# ----------------------

# Basis intervals in semitones mapped to lattice axes
# i-axis: perfect fifth (+7), j-axis: major third (+4), k-axis: minor third (+3)
BASIS = np.array([7, 4, 3], dtype=int)


@dataclass(frozen=True)
class LatticePoint:
    i: int
    j: int
    k: int

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.i, self.j, self.k)


class Tonnetz3D:
    def __init__(self, radius: int = 2, prefer_sharps: bool = True):
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.radius = radius
        self.prefer_sharps = prefer_sharps
        self._nodes: List[Tuple[LatticePoint, int]] = []  # (point, pc)
        self._positions: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
        self._build()

    @staticmethod
    def lattice_pc(i: int, j: int, k: int) -> int:
        # linear combination in semitone space
        return (7 * i + 4 * j + 3 * k) % 12

    def _build(self) -> None:
        r = self.radius
        nodes: List[Tuple[LatticePoint, int]] = []
        positions: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}

        # Slightly skew/scale axes so structure is visually distinct in 3D
        # Use an oblique basis to avoid a plain cube look.
        ei = np.array([1.0, 0.0, 0.0])
        ej = np.array([0.4, math.sqrt(1 - 0.4 ** 2), 0.0])
        ek = np.array([0.3, 0.2, math.sqrt(1 - 0.3 ** 2 - 0.2 ** 2)])

        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                for k in range(-r, r + 1):
                    pc = self.lattice_pc(i, j, k)
                    p = LatticePoint(i, j, k)
                    # embedded position in 3D
                    vec = i * ei + j * ej + k * ek
                    positions[p.to_tuple()] = (float(vec[0]), float(vec[1]), float(vec[2]))
                    nodes.append((p, pc))

        self._nodes = nodes
        self._positions = positions

    # ------------- visualization -------------

    @staticmethod
    def _pc_colors() -> List[str]:
        # 12-color palette (Okabe-Ito + extras) to be distinct on light/dark backgrounds
        return [
            "#E69F00",  # C
            "#56B4E9",  # C#/Db
            "#009E73",  # D
            "#F0E442",  # D#/Eb
            "#0072B2",  # E
            "#D55E00",  # F
            "#CC79A7",  # F#/Gb
            "#999999",  # G
            "#8DD3C7",  # G#/Ab
            "#FB8072",  # A
            "#80B1D3",  # A#/Bb
            "#FDB462",  # B
        ]

    def _scatter_nodes(self) -> go.Scatter3d:
        colors = self._pc_colors()
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        text: List[str] = []
        marker_colors: List[str] = []

        for p, pc in self._nodes:
            x, y, z = self._positions[p.to_tuple()]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            marker_colors.append(colors[pc])
            text.append(f"{pc_to_name(pc, self.prefer_sharps)}\n({p.i},{p.j},{p.k})")

        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            text=text,
            hovertemplate="%{text}",
            marker=dict(size=4, color=marker_colors, opacity=0.95, line=dict(width=0)),
            name="Pitch classes",
            showlegend=False,
        )

    def _edges(self) -> go.Scatter3d:
        # Build axis-aligned edges between neighbor lattice points within radius
        from typing import Optional as _Optional
        xs: List[_Optional[float]] = []
        ys: List[_Optional[float]] = []
        zs: List[_Optional[float]] = []

        r = self.radius
        points = set(self._positions.keys())

        def add_edge(a: Tuple[int, int, int], b: Tuple[int, int, int]):
            ax, ay, az = self._positions[a]
            bx, by, bz = self._positions[b]
            xs.extend([ax, bx, None])
            ys.extend([ay, by, None])
            zs.extend([az, bz, None])

        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                for k in range(-r, r + 1):
                    a = (i, j, k)
                    # edges along +i, +j, +k if neighbor exists
                    for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        b = (i + di, j + dj, k + dk)
                        if b in points:
                            add_edge(a, b)

        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="rgba(120,120,120,0.35)", width=2),
            hoverinfo="none",
            name="Edges",
            showlegend=False,
        )

    # ------------- triad helpers -------------

    @staticmethod
    def notes_for_triad(root: str, quality: str) -> List[int]:
        pc = note_to_pc(root)
        q = quality.strip().lower()
        if q in ("maj", "major", "M"):
            return [pc, (pc + 4) % 12, (pc + 7) % 12]
        if q in ("min", "minor", "m"):
            return [pc, (pc + 3) % 12, (pc + 7) % 12]
        raise ValueError("quality must be one of: maj/major or min/minor")

    def _find_root_anchor(self, root_pc: int, want: str) -> Optional[LatticePoint]:
        # want in {"maj","min"}
        r = self.radius
        candidates: List[Tuple[float, LatticePoint]] = []
        for p, pc in self._nodes:
            if pc == root_pc:
                # check if associated triad nodes would be in-bounds
                if want == "maj":
                    p1 = (p.i + 0, p.j + 1, p.k + 0)  # +M3
                else:
                    p1 = (p.i + 0, p.j + 0, p.k + 1)  # +m3
                p2 = (p.i + 1, p.j + 0, p.k + 0)  # +P5
                if (
                    -r <= p1[0] <= r
                    and -r <= p1[1] <= r
                    and -r <= p1[2] <= r
                    and -r <= p2[0] <= r
                    and -r <= p2[1] <= r
                    and -r <= p2[2] <= r
                ):
                    # distance to center to pick the most central realization
                    d = math.sqrt(p.i**2 + p.j**2 + p.k**2)
                    candidates.append((d, p))
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1] if candidates else None

    def _highlight_triad_traces(self, root: str, quality: str) -> List[go.Scatter3d]:
        root_pc = note_to_pc(root)
        # normalize quality
        qn = quality.strip().lower()
        if qn in ("maj", "major"):
            want = "maj"
        elif qn in ("min", "minor", "m"):
            want = "min"
        else:
            raise ValueError("quality must be one of: maj/major or min/minor/m")

        anchor = self._find_root_anchor(root_pc, want)
        if anchor is None:
            return []

        if want == "maj":
            p_root = anchor.to_tuple()
            p_third = (anchor.i, anchor.j + 1, anchor.k)
            p_fifth = (anchor.i + 1, anchor.j, anchor.k)
        else:
            p_root = anchor.to_tuple()
            p_third = (anchor.i, anchor.j, anchor.k + 1)
            p_fifth = (anchor.i + 1, anchor.j, anchor.k)

        pts = [p_root, p_third, p_fifth]
        names = [pc_to_name(x, self.prefer_sharps) for x in self.notes_for_triad(root, want)]

        # Highlight points
        xs = [self._positions[p][0] for p in pts]
        ys = [self._positions[p][1] for p in pts]
        zs = [self._positions[p][2] for p in pts]

        triad_points = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            text=names,
            textposition="top center",
            textfont=dict(size=12, color="#222"),
            marker=dict(size=9, color="#FF4136", symbol="diamond", line=dict(color="#111", width=1)),
            name=f"Triad: {root} {quality}",
            showlegend=True,
        )

        # Triangle edges
        from typing import Optional as _Optional
        line_x: List[_Optional[float]] = [xs[0], xs[1], None, xs[1], xs[2], None, xs[2], xs[0], None]
        line_y: List[_Optional[float]] = [ys[0], ys[1], None, ys[1], ys[2], None, ys[2], ys[0], None]
        line_z: List[_Optional[float]] = [zs[0], zs[1], None, zs[1], zs[2], None, zs[2], zs[0], None]

        triad_edges = go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line=dict(color="#FF4136", width=5),
            name="Triad edges",
            showlegend=False,
        )

        return [triad_points, triad_edges]

    # ------------- public API -------------

    def figure(self, highlight: Optional[Tuple[str, str]] = None) -> go.Figure:
        data = [self._edges(), self._scatter_nodes()]
        if highlight is not None:
            root, quality = highlight
            data.extend(self._highlight_triad_traces(root, quality))

        layout = go.Layout(
            title=f"3D Tonnetz (radius={self.radius})",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        )

        return go.Figure(data=data, layout=layout)

    def save_html(self, path: str, highlight: Optional[Tuple[str, str]] = None) -> None:
        fig = self.figure(highlight)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)

    def show(self, highlight: Optional[Tuple[str, str]] = None) -> None:
        fig = self.figure(highlight)
        fig.show()


# small public helper

def notes_for_triad(root: str, quality: str) -> List[str]:
    pcs = Tonnetz3D.notes_for_triad(root, quality)
    return [pc_to_name(pc) for pc in pcs]
