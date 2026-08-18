"""Animated BWT decompression by the last-to-first property, lecture style.

Follows Read_Mapping.pptx's "Efficient BWT Decompression" slides: rank every
symbol in the first and last columns of the sorted rotation matrix, then walk
from row to row. The kth occurrence of a symbol in the last column is the kth
occurrence of that symbol in the first column, so each step names the previous
character of the text and where to jump next. The text assembles around a circle.

The walk is asserted to visit every row once and to spell the text exactly.

Run:  python3 example_bwt_walk.py OUTPUT.gif
"""

import math
import os
import sys
import tempfile

import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, BLUE, DIM, FAINT, INK, MONO, MONO_BOLD,
                           OPTIMA_ITALIC, RED, new_axes)
from make_gif import assemble_transparent_gif

TEXT = "panamabananas$"
PUBLISHED_BWT = "smnpbnnaaaaa$a"

FIGURE_WIDTH = 10.4
FIGURE_HEIGHT = 6.2
RENDER_DPI = 130
OUTPUT_WIDTH = 1352
OUTPUT_HEIGHT = 806

MATRIX_LEFT = 0.55
MATRIX_TOP = 5.30
CHAR_ADVANCE = 0.205
ROW_HEIGHT = 0.345
CHAR_SIZE = 16.0
RANK_SIZE = 10.0
RANK_DX = 0.092
RANK_DY = -0.090

# Measured from Read_Mapping.pptx slides 358-371: the unknown middle of each row is
# A6A6A6, the symbol under discussion is pure red, and settled symbols are black.
DECK_GREY = "#A6A6A6"
DECK_RED = "#FF0000"

CIRCLE_X = 7.55
CIRCLE_Y = 2.90
LETTER_RADIUS = 1.85
ARC_RADIUS = 1.02
CIRCLE_SIZE = 21.0

CAPTION_Y = 5.86
READOUT_Y = 0.42
READOUT_SIZE = 19.0
LABEL_SIZE = 15.0

# Phillip: the walk was too fast to follow. Each step now gets time to be read,
# and the dwell decays as the move becomes familiar, so the animation slows down
# where the idea is new and speeds up where it is repetition.
STEP_MS = 2600
STEP_FLOOR_MS = 1400
STEP_DECAY_MS = 120
JUMP_MS = 1600
JUMP_FLOOR_MS = 900
JUMP_DECAY_MS = 70
FIRST_HOLD_MS = 4200
FINAL_HOLD_MS = 5600


def sorted_matrix() -> "list[str]":
    rotations = []
    offset = 0
    while offset < len(TEXT):
        rotations.append(TEXT[offset:] + TEXT[:offset])
        offset = offset + 1
    rotations.sort()
    return rotations


MATRIX = sorted_matrix()
SIZE = len(TEXT)
FIRST = "".join(row[0] for row in MATRIX)
LAST = "".join(row[SIZE - 1] for row in MATRIX)
CYCLIC = TEXT[SIZE - 1] + TEXT[0:SIZE - 1]


def ranks(column: str) -> "list[int]":
    """Occurrence number of each character within its own symbol."""
    seen = {}
    result = []
    position = 0
    while position < len(column):
        symbol = column[position]
        seen[symbol] = seen.get(symbol, 0) + 1
        result.append(seen[symbol])
        position = position + 1
    return result


FIRST_RANK = ranks(FIRST)
LAST_RANK = ranks(LAST)


def first_row_of(symbol: str, rank: int) -> int:
    row = 0
    while row < SIZE:
        if FIRST[row] == symbol and FIRST_RANK[row] == rank:
            return row
        row = row + 1
    raise ValueError("no row for %s%d" % (symbol, rank))


def walk() -> "list[dict]":
    """Each step: the row we are on, the symbol it yields, and where we jump."""
    steps = []
    row = 0
    count = 0
    while count < SIZE:
        symbol = LAST[row]
        rank = LAST_RANK[row]
        target = first_row_of(symbol, rank)
        steps.append({"row": row, "symbol": symbol, "rank": rank,
                      "target": target, "slot": SIZE - 1 - count})
        row = target
        count = count + 1
    return steps


STEPS = walk()


def verify() -> "list[str]":
    lines = []
    assert LAST == PUBLISHED_BWT, (
        "computed last column %r but the slide prints %r" % (LAST, PUBLISHED_BWT))
    lines.append("last column is %s, matching the lecture slide" % LAST)

    assert FIRST == "".join(sorted(TEXT)), "first column must be the sorted text"
    lines.append("first column is the sorted text, %s" % FIRST)

    visited = []
    for step in STEPS:
        visited.append(step["row"])
    assert sorted(visited) == list(range(SIZE)), (
        "the walk must visit every row exactly once, got %s" % sorted(visited))
    assert STEPS[SIZE - 1]["target"] == 0, "the walk must close back on row 0"
    lines.append("the walk visits all %d rows once and closes on row 0" % SIZE)

    for step in STEPS:
        target = step["target"]
        assert FIRST[target] == step["symbol"], "jump landed on the wrong symbol"
        assert FIRST_RANK[target] == step["rank"], "jump landed on the wrong rank"
    lines.append("every jump lands on the same symbol and rank in the first column")

    emitted = []
    for step in STEPS:
        emitted.append(step["symbol"])
    emitted.reverse()
    assert "".join(emitted) == CYCLIC, (
        "the walk spelled %r, not %r" % ("".join(emitted), CYCLIC))
    lines.append("read backwards the walk spells %s" % CYCLIC)

    rebuilt = ""
    for step in STEPS:
        rebuilt = step["symbol"] + rebuilt
    cut = rebuilt.index("$")
    rotated = rebuilt[cut + 1:] + rebuilt[0:cut + 1]
    assert rotated == TEXT, "rotating to the sentinel gave %r" % rotated
    lines.append("rotating to the sentinel recovers %s" % TEXT)

    for step in STEPS:
        assert CYCLIC[step["slot"]] == step["symbol"], (
            "step fills circle slot %d with the wrong symbol" % step["slot"])
    lines.append("each step fills the circle slot its symbol belongs in")

    width = SIZE * CHAR_ADVANCE
    lowest = MATRIX_TOP - (SIZE - 1) * ROW_HEIGHT
    assert lowest > 0.32, "matrix runs off the bottom"
    circle_bottom = CIRCLE_Y - LETTER_RADIUS
    assert READOUT_Y + 0.28 < circle_bottom, "readout collides with the circle"
    assert MATRIX_LEFT + width < CIRCLE_X - LETTER_RADIUS - 0.4, (
        "matrix collides with the circle")
    assert CIRCLE_X + LETTER_RADIUS + 0.3 < FIGURE_WIDTH, "circle runs off the right"
    lines.append("matrix %.2f in wide, clear of the circle at x=%.1f" % (width, CIRCLE_X))
    return lines


def slot_angle(slot: int) -> float:
    """Slot 0 sits at the top; slots advance clockwise."""
    return math.radians(90.0 - slot * (360.0 / SIZE))


def slot_xy(slot: int, radius: float) -> "tuple[float, float]":
    angle = slot_angle(slot)
    return (CIRCLE_X + radius * math.cos(angle), CIRCLE_Y + radius * math.sin(angle))


def scheduled(base: int, floor: int, decay: int, index: int) -> int:
    """Dwell that shortens as a repeated step becomes familiar.

    GIF stores delays in centiseconds, so the result is rounded to 10 ms or the
    real playback runs shorter than the script reports.
    """
    value = base - decay * index
    if value < floor:
        value = floor
    return int(round(value / 10.0)) * 10


def base_frame(duration: int) -> dict:
    return {"current": -1, "target": -1, "known": set(), "caption": "",
            "readout": False, "duration_ms": duration}


def build_specs() -> "list[dict]":
    specs = []
    known = set([0])

    opening = base_frame(FIRST_HOLD_MS)
    opening["known"] = set(known)
    opening["current"] = 0
    opening["caption"] = ("The kth %s in the last column is the kth %s in the first"
                          % ("symbol", "symbol"))
    specs.append(opening)

    index = 0
    while index < SIZE - 1:
        step = STEPS[index]
        known.add(step["slot"])

        reveal = base_frame(scheduled(STEP_MS, STEP_FLOOR_MS,
                                      STEP_DECAY_MS, index))
        reveal["current"] = step["row"]
        reveal["known"] = set(known)
        reveal["caption"] = ("Row %d ends in %s%d, so the previous character is %s"
                             % (step["row"], step["symbol"], step["rank"],
                                step["symbol"]))
        specs.append(reveal)

        jump = base_frame(scheduled(JUMP_MS, JUMP_FLOOR_MS,
                                    JUMP_DECAY_MS, index))
        jump["current"] = step["row"]
        jump["target"] = step["target"]
        jump["known"] = set(known)
        jump["caption"] = ("Jump to the row starting with %s%d"
                           % (step["symbol"], step["rank"]))
        specs.append(jump)
        index = index + 1

    final = base_frame(FINAL_HOLD_MS)
    slot = 0
    while slot < SIZE:
        known.add(slot)
        slot = slot + 1
    final["known"] = set(known)
    final["current"] = STEPS[SIZE - 2]["target"]
    final["caption"] = "Every character recovered from the last column alone"
    final["readout"] = True
    specs.append(final)
    return specs


def draw_symbol(axis: "plt.Axes", x: float, y: float, symbol: str, rank: int,
                colour: str) -> None:
    axis.text(x, y, symbol, fontsize=CHAR_SIZE, color=colour, ha="center",
              va="center", fontproperties=MONO_BOLD, zorder=5)
    axis.text(x + RANK_DX, y + RANK_DY, str(rank), fontsize=RANK_SIZE,
              color=colour, ha="center", va="center", fontproperties=MONO,
              zorder=5)


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)

    row = 0
    while row < SIZE:
        y = MATRIX_TOP - row * ROW_HEIGHT
        if row == spec["current"]:
            first_colour = INK
            last_colour = DECK_RED
        elif row == spec["target"]:
            first_colour = DECK_RED
            last_colour = INK
        else:
            first_colour = INK
            last_colour = INK
        draw_symbol(axis, MATRIX_LEFT + 0.5 * CHAR_ADVANCE, y, FIRST[row],
                    FIRST_RANK[row], first_colour)
        column = 1
        while column < SIZE - 1:
            axis.text(MATRIX_LEFT + (column + 0.5) * CHAR_ADVANCE, y,
                      MATRIX[row][column], fontsize=CHAR_SIZE, color=DECK_GREY,
                      ha="center", va="center", fontproperties=MONO, zorder=4)
            column = column + 1
        draw_symbol(axis, MATRIX_LEFT + (SIZE - 0.5) * CHAR_ADVANCE, y, LAST[row],
                    LAST_RANK[row], last_colour)
        row = row + 1

    # the reconstruction direction, drawn as an arc inside the ring
    angles = []
    count = 0
    while count <= 60:
        angles.append(math.radians(-60.0 + count * 4.6))
        count = count + 1
    xs = []
    ys = []
    for angle in angles:
        xs.append(CIRCLE_X + ARC_RADIUS * math.cos(angle))
        ys.append(CIRCLE_Y + ARC_RADIUS * math.sin(angle))
    axis.plot(xs, ys, color=INK, linewidth=1.6, zorder=3)
    # A (3, 0, rot) marker is a triangle pointing up, rotated counterclockwise by
    # rot, so tangent direction a + 90 needs rot = a.
    head_angle = angles[len(angles) - 1]
    axis.plot([xs[len(xs) - 1]], [ys[len(ys) - 1]],
              marker=(3, 0, math.degrees(head_angle)), markersize=11, color=INK,
              zorder=3)

    slot = 0
    while slot < SIZE:
        x, y = slot_xy(slot, LETTER_RADIUS)
        if slot in spec["known"]:
            colour = INK
        else:
            colour = FAINT
        if spec["current"] >= 0 and not spec["readout"]:
            live = -1
            for step in STEPS:
                if step["row"] == spec["current"]:
                    live = step["slot"]
            if slot == live and slot in spec["known"]:
                colour = DECK_RED
        axis.text(x, y, CYCLIC[slot], fontsize=CIRCLE_SIZE, color=colour,
                  ha="center", va="center", fontproperties=MONO_BOLD, zorder=5)
        slot = slot + 1

    if spec["caption"] != "":
        axis.text(FIGURE_WIDTH / 2, CAPTION_Y, spec["caption"], fontsize=LABEL_SIZE,
                  color=INK, ha="center", va="center", fontproperties=OPTIMA_ITALIC,
                  zorder=7)

    if spec["readout"]:
        axis.text(CIRCLE_X, READOUT_Y, TEXT, fontsize=READOUT_SIZE, color=INK,
                  ha="center", va="center", fontproperties=MONO_BOLD, zorder=7)

    figure.savefig(output_path, transparent=True)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: example_bwt_walk.py OUTPUT.gif")
        return
    print("Structural checks:")
    for line in verify():
        print("  ok: " + line)

    specs = build_specs()
    durations = []
    for spec in specs:
        durations.append(spec["duration_ms"])
    print("Rendering %d frames at %dx%d, %.1f s of playback..."
          % (len(specs), OUTPUT_WIDTH, OUTPUT_HEIGHT, sum(durations) / 1000.0))
    directory = tempfile.mkdtemp(prefix="bwt_walk_")
    frame_paths = []
    index = 0
    while index < len(specs):
        path = os.path.join(directory, "frame_%04d.png" % index)
        draw_frame(specs[index], path)
        frame_paths.append(path)
        index = index + 1

    assemble_transparent_gif(frame_paths, sys.argv[1], width=OUTPUT_WIDTH,
                             height=OUTPUT_HEIGHT, frame_durations=durations)
    print("Saved " + sys.argv[1])


if __name__ == "__main__":
    main()
