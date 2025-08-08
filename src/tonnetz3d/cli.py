import argparse
from pathlib import Path
from typing import Optional, Tuple

from .tonnetz import Tonnetz3D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D Tonnetz visualizer (Plotly)")
    p.add_argument("--radius", type=int, default=2, help="Lattice radius (>=1)")
    p.add_argument("--root", type=str, default="C", help="Root note (e.g., C, F#, Eb)")
    p.add_argument("--quality", type=str, default="maj", help="Triad quality: maj|min")
    p.add_argument("--no-sharps", action="store_true", help="Prefer flats in labels")
    p.add_argument("--show", action="store_true", help="Open interactive figure window")
    p.add_argument("--out", type=str, default="tonnetz.html", help="Output HTML file path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prefer_sharps = not args.no_sharps
    t = Tonnetz3D(radius=max(1, int(args.radius)), prefer_sharps=prefer_sharps)

    highlight: Optional[Tuple[str, str]] = (args.root, args.quality) if args.root and args.quality else None

    out_path = Path(args.out)
    if args.show:
        t.show(highlight)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        t.save_html(str(out_path), highlight)
        print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
