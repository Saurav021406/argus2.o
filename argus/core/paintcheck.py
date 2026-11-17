"""Automatic check for the Paint tasks: is there ink on the white canvas, and does it look like the request?
A line = a long stroke. A zigzag = a wide, tall stroke that also changes direction at least twice.

Finds the biggest white region of the screen (the canvas), then measures the dark pixels inside it.
Heuristic and tuned for a default Paint window: verify it on YOUR screen first with
    python -m argus.core inkcheck
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

from PIL import Image

BLOCK = 8


def turning_points(mask: Image.Image, bbox, min_swing: int = 25) -> int:
    """How many times does the stroke change vertical direction (by at least min_swing px)?
    Mean ink height per column: a straight line is monotone (0 turns); a 3-stroke zigzag has 2."""
    x0, y0, x1, y1 = bbox
    px = mask.load()
    heights = []
    for x in range(x0, x1):
        ys = [y for y in range(y0, y1) if px[x, y]]
        if ys:
            heights.append(sum(ys) / len(ys))
    if len(heights) < 2:
        return 0
    turns, extreme, direction = 0, heights[0], 0
    for h in heights[1:]:
        if direction >= 0 and h > extreme:
            extreme = h
            direction = 1 if h - heights[0] >= min_swing or direction == 1 else direction
        elif direction <= 0 and h < extreme:
            extreme = h
            direction = -1 if heights[0] - h >= min_swing or direction == -1 else direction
        elif direction == 1 and extreme - h >= min_swing:
            turns, direction, extreme = turns + 1, -1, h
        elif direction == -1 and h - extreme >= min_swing:
            turns, direction, extreme = turns + 1, 1, h
    return turns


def ink_stats(img: Image.Image, white: int = 235, dark: int = 110) -> Dict:
    g = img.convert("L")
    W, H = g.size
    small = g.resize((max(1, W // BLOCK), max(1, H // BLOCK)), Image.BOX)
    sw, sh = small.size
    px = small.load()
    seen = [[False] * sw for _ in range(sh)]
    best: Optional[Tuple[int, int, int, int, int]] = None          # area, x0, y0, x1, y1 (in blocks)
    for sy in range(sh):
        for sx in range(sw):
            if seen[sy][sx] or px[sx, sy] < white:
                continue
            q, area = deque([(sx, sy)]), 0
            seen[sy][sx] = True
            x0 = x1 = sx
            y0 = y1 = sy
            while q:
                x, y = q.popleft()
                area += 1
                x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < sw and 0 <= ny < sh and not seen[ny][nx] and px[nx, ny] >= white:
                        seen[ny][nx] = True
                        q.append((nx, ny))
            if best is None or area > best[0]:
                best = (area, x0, y0, x1, y1)
    bright_pct = round(100.0 * (best[0] * BLOCK * BLOCK if best else 0) / (W * H), 1)   # biggest white region, % of screen
    if best is None or best[0] * BLOCK * BLOCK < 0.04 * W * H:
        return {"canvas": None, "ink_px": 0, "ink_w": 0, "ink_h": 0, "bright_pct": bright_pct}
    _, bx0, by0, bx1, by1 = best
    box = (bx0 * BLOCK, by0 * BLOCK, min(W, (bx1 + 1) * BLOCK), min(H, (by1 + 1) * BLOCK))
    canvas = g.crop(box)
    mask = canvas.point(lambda v: 255 if v < dark else 0)
    bb = mask.getbbox()
    ink_px = mask.histogram()[255]
    return {"canvas": box, "ink_px": ink_px, "ink_w": (bb[2] - bb[0]) if bb else 0,
            "ink_h": (bb[3] - bb[1]) if bb else 0, "ink_bbox": bb,
            "turns": turning_points(mask, bb) if bb else 0, "bright_pct": bright_pct}


def judge_paint(stats: Dict, kind: str) -> Tuple[bool, str]:
    """kind: 'line' needs a long stroke; 'zigzag' also has to wander in both directions."""
    if stats["canvas"] is None:
        return False, "no white canvas found (Paint not open?)"
    if stats["ink_px"] < 80:
        return False, f"almost no ink ({stats['ink_px']} px)"
    w, h = stats["ink_w"], stats["ink_h"]
    if kind == "line":
        ok = max(w, h) >= 150
        return ok, f"ink {w}x{h}px" + ("" if ok else " (too short for a line, need >=150)")
    turns = stats.get("turns", 0)
    ok = w >= 150 and h >= 60 and turns >= 2
    return ok, f"ink {w}x{h}px, {turns} direction changes" + ("" if ok else " (a zigzag needs >=150 wide, >=60 tall and >=2 turns)")
