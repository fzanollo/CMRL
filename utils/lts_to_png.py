"""Render Sokoban board PNGs from `.lts` files or folders of `.lts` files."""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageDraw


def parse_diagram(lines: Sequence[str]) -> List[List[str]]:
    """Read the board diagram comment block into a 2D token grid."""
    in_block = False
    rows: Dict[int, List[str]] = {}
    for line in lines:
        if not in_block:
            if line.strip().startswith("//") and "board diagram" in line.lower():
                in_block = True
            continue

        # stop when we hit a non-comment config line (const, set, etc.)
        if not line.strip().startswith("//"):
            # allow a few non-grid comment lines like //// B = Brick by continuing
            if re.match(r"\s*///*\s*[BFOb\-\s=]", line):
                # still a comment; continue parsing
                pass
            else:
                break

        # match lines like: //1 - - - - -  (leading // then row number)
        m = re.match(r"\s*//\s*(\d+)\s+(.*)$", line)
        if m:
            rnum = int(m.group(1))
            rest = m.group(2).strip()
            if rest:
                tokens = [tok for tok in re.split(r"\s+", rest) if tok != ""]
                rows[rnum] = tokens
            continue

        # match header line like: //  1 2 3 4 5  (ignore)
        # other comment lines are ignored

    if not rows:
        raise ValueError("No board diagram rows found in the input .lts file")

    # produce ordered list
    ordered = [rows[k] for k in sorted(rows.keys())]
    # normalize row lengths
    width = max(len(r) for r in ordered)
    normalized = [list(r + ["-"] * (width - len(r))) for r in ordered]
    return normalized


def render_board_using_sprites(
    board: List[List[str]],
    sprites_dir: Path,
    use_tiled_bg: bool = False,
 ) -> Image.Image:
    sprites: Dict[str, Optional[Image.Image]] = {
        name: Image.open(sprites_dir / f"{name}.png").convert("RGBA") if (sprites_dir / f"{name}.png").exists() else None
        for name in ("B", "F", "O", "player")
    }
    tile = max((max(img.width, img.height) for img in sprites.values() if img is not None), default=32)

    rows = len(board)
    cols = len(board[0]) if rows else 0

    grid_width = 3
    out = Image.new("RGBA", ((cols * tile)+grid_width, (rows * tile)+grid_width), (0, 0, 0, 0))

    draw = ImageDraw.Draw(out)

    empty_path = sprites_dir / "empty.png"
    empty_sprite = Image.open(empty_path).convert("RGBA") if empty_path.exists() else None

    for r in range(rows):
        for c in range(cols):
            px = c * tile
            py = r * tile
            if use_tiled_bg and empty_sprite is not None:
                out.paste(empty_sprite, (px, py), empty_sprite)
            else:
                draw.rectangle([px, py, px + tile - 1, py + tile - 1], fill=(255, 255, 255, 255))
            token = board[r][c].upper()
            if token == "-":
                continue
            if token == "B":
                spr = sprites.get("B")
            elif token == "F":
                spr = sprites.get("F")
            elif token == "O":
                spr = sprites.get("O")
            else:
                spr = None

            if spr is not None:
                # center sprite if different size
                sw, sh = spr.size
                ox = px + (tile - sw) // 2
                oy = py + (tile - sh) // 2
                out.paste(spr, (ox, oy), spr)

    ir, ic = 0, 0
    if 0 <= ir < rows and 0 <= ic < cols:
        pspr = sprites.get("player")
        if pspr is not None:
            sw, sh = pspr.size
            ox = ic * tile + (tile - sw) // 2
            oy = ir * tile + (tile - sh) // 2
            out.paste(pspr, (ox, oy), pspr)
        else:
            print("WARNING: player sprite not found; skipped")
    grid_color = (200, 200, 200, 255)
    w = cols * tile
    h = rows * tile
    for i in range(cols + 1):
        x = i * tile
        draw.line((x, 0, x, h), fill=grid_color, width=grid_width)
    for j in range(rows + 1):
        y = j * tile
        draw.line((0, y, w, y), fill=grid_color, width=grid_width)

    return out


def render_lts_file(lts_path: Path, sprites_dir: Path, use_tiled_bg: bool) -> Path:
    lines = lts_path.read_text(encoding="utf-8").splitlines()
    board = parse_diagram(lines)
    img = render_board_using_sprites(board, sprites_dir, use_tiled_bg=use_tiled_bg)
    out_path = lts_path.with_suffix(".png")
    img.save(out_path)
    print(f"Saved image to: {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Render Sokoban board diagrams to PNG using sprites in utils/sprites/")
    p.add_argument("path", type=Path, nargs="?", default=Path("lts/sokoban-4-1/sokoban-4-1_1obstacleC1-1.lts"), help="an .lts file or a folder containing .lts files (default: sample file)")
    p.add_argument("--use-tiled-bg", action="store_true", default=True, help="use sprites/empty.png for empty cells (default: True)")
    p.add_argument("--no-tiled-bg", action="store_false", dest="use_tiled_bg", help="do not use empty.png; use white background instead")
    args = p.parse_args()

    sprites_dir = Path(__file__).resolve().with_name("sprites")
    path = args.path
    if path.is_dir():
        lts_files = sorted(path.rglob("*.lts"))
        if not lts_files:
            print(f"No .lts files found in folder: {path}")
            return
        print(f"Found {len(lts_files)} .lts files in: {path}")
        for lts_file in lts_files:
            print(f"Reading LTS file: {lts_file}")
            render_lts_file(lts_file, sprites_dir, args.use_tiled_bg)
        return

    print(f"Reading LTS file: {path}")
    render_lts_file(path, sprites_dir, args.use_tiled_bg)


if __name__ == "__main__":
    main()
