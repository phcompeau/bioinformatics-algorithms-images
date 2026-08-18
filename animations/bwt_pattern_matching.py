"""Animated backward search with the BWT, in the 02-180 lecture style.

Follows Read_Mapping.pptx's "We Match Patterns Backward" slides: only the first
and last columns of the sorted rotation matrix are known, and the pattern is
matched right to left. At each step the rows whose last column carries the next
pattern symbol are mapped, by the first-last property, to a contiguous band of
rows in the first column. The band that survives holds every occurrence.

Everything is asserted: the band is contiguous at every step, its size equals
the true number of occurrences, and the reported positions are exactly where the
pattern really occurs in the text.

Run:  python3 example_bwt_matching.py OUTPUT.gif
"""

import os
import sys
import tempfile

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, DIM, FAINT, INK, MONO, MONO_BOLD, RED,
                           OPTIMA_ITALIC, new_axes)
from make_gif import assemble_transparent_gif

TEXT = "panamabananas$"
PATTERN = "ana"
PUBLISHED_OCCURRENCES = 3

FIGURE_WIDTH = 8.2
FIGURE_HEIGHT = 6.8
RENDER_DPI = 150
OUTPUT_WIDTH = 1230
OUTPUT_HEIGHT = 1020

CHAR_ADVANCE = 0.235
ROW_HEIGHT = 0.335
CHAR_SIZE = 17.0
RANK_SIZE = 10.0
RANK_DX = 0.095
RANK_DY = -0.095
MATRIX_TOP = 5.72

READOUT_Y = 0.80
READOUT_SIZE = 24.0
BAND_PAD = 0.14
# Phillip: every step needs a line of explanation, or the band just moves.
CAPTION_Y = 6.42
LABEL_SIZE = 15.0
OPENING_MS = 3400
LOOK_MS = 2800
LANDED_MS = 2600
FINAL_MS = 5600


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


def ranks(column: str) -> "list[int]":
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


def suffix_array() -> "list[int]":
    pairs = []
    start = 0
    while start < SIZE:
        pairs.append((TEXT[start:], start))
        start = start + 1
    pairs.sort()
    result = []
    for suffix, start in pairs:
        result.append(start)
    return result


SUFFIX_ARRAY = suffix_array()


def backward_search() -> "list[dict]":
    """One entry per pattern symbol, right to left."""
    top = 0
    bottom = SIZE - 1
    steps = []
    index = len(PATTERN) - 1
    while index >= 0:
        symbol = PATTERN[index]
        hits = []
        row = top
        while row <= bottom:
            if LAST[row] == symbol:
                hits.append(row)
            row = row + 1
        if len(hits) == 0:
            steps.append({"symbol": symbol, "hits": [], "top": top,
                          "bottom": bottom, "new_top": -1, "new_bottom": -1,
                          "matched": PATTERN[index:]})
            break
        new_top = first_row_of(symbol, LAST_RANK[hits[0]])
        new_bottom = first_row_of(symbol, LAST_RANK[hits[len(hits) - 1]])
        steps.append({"symbol": symbol, "hits": list(hits), "top": top,
                      "bottom": bottom, "new_top": new_top,
                      "new_bottom": new_bottom, "matched": PATTERN[index:]})
        top = new_top
        bottom = new_bottom
        index = index - 1
    return steps


STEPS = backward_search()
FINAL_TOP = STEPS[len(STEPS) - 1]["new_top"]
FINAL_BOTTOM = STEPS[len(STEPS) - 1]["new_bottom"]


def occurrences() -> "list[int]":
    result = []
    row = FINAL_TOP
    while row <= FINAL_BOTTOM:
        result.append(SUFFIX_ARRAY[row])
        row = row + 1
    result.sort()
    return result


OCCURRENCES = occurrences()


def verify() -> "list[str]":
    lines = []

    assert len(STEPS) == len(PATTERN), "one step per pattern symbol"
    lines.append("backward search took %d steps, one per symbol of %s"
                 % (len(STEPS), PATTERN))

    for step in STEPS:
        assert len(step["hits"]) > 0, "pattern unexpectedly absent"
        walker = 1
        while walker < len(step["hits"]):
            assert step["hits"][walker] > step["hits"][walker - 1], "hits unsorted"
            walker = walker + 1
        span = step["new_bottom"] - step["new_top"] + 1
        assert span == len(step["hits"]), (
            "band of %d rows does not match %d hits" % (span, len(step["hits"])))
        row = step["new_top"]
        while row <= step["new_bottom"]:
            assert FIRST[row] == step["symbol"], (
                "band row %d does not start with %s" % (row, step["symbol"]))
            row = row + 1
    lines.append("every step maps its hits onto a contiguous band of the first column")

    for step in STEPS:
        row = step["new_top"]
        while row <= step["new_bottom"]:
            assert MATRIX[row].startswith(step["matched"]), (
                "row %d does not start with %s" % (row, step["matched"]))
            row = row + 1
        outside = 0
        while outside < SIZE:
            inside = step["new_top"] <= outside <= step["new_bottom"]
            if not inside:
                assert not MATRIX[outside].startswith(step["matched"]), (
                    "row %d starts with %s but is outside the band"
                    % (outside, step["matched"]))
            outside = outside + 1
    lines.append("each band holds exactly the rows beginning with the matched suffix")

    assert len(OCCURRENCES) == PUBLISHED_OCCURRENCES, (
        "found %d occurrences, the slides say %d"
        % (len(OCCURRENCES), PUBLISHED_OCCURRENCES))
    for position in OCCURRENCES:
        assert TEXT[position:position + len(PATTERN)] == PATTERN, (
            "position %d does not spell %s" % (position, PATTERN))
    brute = []
    start = 0
    while start + len(PATTERN) <= len(TEXT):
        if TEXT[start:start + len(PATTERN)] == PATTERN:
            brute.append(start)
        start = start + 1
    assert OCCURRENCES == brute, (
        "search found %s but scanning the text finds %s" % (OCCURRENCES, brute))
    lines.append("%s occurs at %s, agreeing with a brute-force scan"
                 % (PATTERN, " ".join(str(value) for value in OCCURRENCES)))

    lowest = MATRIX_TOP - (SIZE - 1) * ROW_HEIGHT
    assert lowest > READOUT_Y + 0.4, "matrix collides with the readout"
    assert MATRIX_TOP + 0.3 < FIGURE_HEIGHT, "matrix runs off the top"
    lines.append("matrix spans %.2f to %.2f in, clear of the readout"
                 % (lowest, MATRIX_TOP))
    return lines


LEFT = (FIGURE_WIDTH - SIZE * CHAR_ADVANCE) / 2


def char_x(column: int) -> float:
    return LEFT + (column + 0.5) * CHAR_ADVANCE


def row_y(row: int) -> float:
    return MATRIX_TOP - row * ROW_HEIGHT


def base_frame(duration: int) -> dict:
    return {"top": 0, "bottom": SIZE - 1, "hits": [], "symbol": "",
            "matched": "", "positions": False, "caption": "",
            "duration_ms": duration}


def build_specs() -> "list[dict]":
    specs = []

    opening = base_frame(OPENING_MS)
    opening["caption"] = ("Only the first and last columns are known, so %s is "
                          "matched from the right" % PATTERN)
    specs.append(opening)

    rule = base_frame(OPENING_MS)
    rule["caption"] = ("Last-to-first: the kth copy of a symbol in the last column "
                       "is the kth copy of it in the first")
    specs.append(rule)

    for step in STEPS:
        look = base_frame(LOOK_MS)
        look["top"] = step["top"]
        look["bottom"] = step["bottom"]
        look["hits"] = list(step["hits"])
        look["symbol"] = step["symbol"]
        look["matched"] = step["matched"][1:]
        look["caption"] = ("%d rows of the band end in %s"
                           % (len(step["hits"]), step["symbol"]))
        specs.append(look)

        landed = base_frame(LANDED_MS)
        landed["top"] = step["new_top"]
        landed["bottom"] = step["new_bottom"]
        landed["symbol"] = step["symbol"]
        landed["matched"] = step["matched"]
        landed["caption"] = ("By the last-to-first property those same %s's start "
                             "a band of rows, which now matches %s"
                             % (step["symbol"], step["matched"]))
        specs.append(landed)

    final = base_frame(FINAL_MS)
    final["top"] = FINAL_TOP
    final["bottom"] = FINAL_BOTTOM
    final["matched"] = PATTERN
    final["positions"] = True
    final["caption"] = ("Every row left in the band is an occurrence of %s" % PATTERN)
    specs.append(final)
    return specs


def draw_symbol(axis: "plt.Axes", x: float, y: float, symbol: str, rank: int,
                colour: str, bold: bool) -> None:
    if bold:
        font = MONO_BOLD
    else:
        font = MONO
    axis.text(x, y, symbol, fontsize=CHAR_SIZE, color=colour, ha="center",
              va="center", fontproperties=font, zorder=6)
    axis.text(x + RANK_DX, y + RANK_DY, str(rank), fontsize=RANK_SIZE,
              color=colour, ha="center", va="center", fontproperties=MONO,
              zorder=6)


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)

    band = patches.Rectangle(
        (LEFT - BAND_PAD, row_y(spec["bottom"]) - ROW_HEIGHT / 2),
        SIZE * CHAR_ADVANCE + 2 * BAND_PAD,
        (spec["bottom"] - spec["top"] + 1) * ROW_HEIGHT,
        facecolor="#E2DCCB", edgecolor="none", zorder=2)
    axis.add_patch(band)

    row = 0
    while row < SIZE:
        y = row_y(row)
        inside = spec["top"] <= row <= spec["bottom"]
        if inside:
            base = INK
        else:
            base = FAINT
        if row in spec["hits"]:
            last_colour = RED
            bold = True
        elif inside:
            last_colour = INK
            bold = False
        else:
            last_colour = FAINT
            bold = False
        draw_symbol(axis, char_x(0), y, FIRST[row], FIRST_RANK[row], base, False)
        column = 1
        while column < SIZE - 1:
            axis.text(char_x(column), y, "?", fontsize=CHAR_SIZE,
                      color=FAINT, ha="center", va="center", fontproperties=MONO,
                      zorder=5)
            column = column + 1
        draw_symbol(axis, char_x(SIZE - 1), y, LAST[row], LAST_RANK[row],
                    last_colour, bold)
        row = row + 1

    if spec["caption"] != "":
        axis.text(FIGURE_WIDTH / 2, CAPTION_Y, spec["caption"], fontsize=LABEL_SIZE,
                  color=INK, ha="center", va="center", fontproperties=OPTIMA_ITALIC,
                  zorder=7)

    matched = spec["matched"]
    unmatched = PATTERN[0:len(PATTERN) - len(matched)]
    advance = READOUT_SIZE / 72.0 * 0.60
    total = len(PATTERN)
    x = FIGURE_WIDTH / 2 - total * advance / 2
    axis.text(x, READOUT_Y, unmatched, fontsize=READOUT_SIZE, color=FAINT,
              ha="left", va="center", fontproperties=MONO_BOLD, zorder=7)
    axis.text(x + len(unmatched) * advance, READOUT_Y, matched,
              fontsize=READOUT_SIZE, color=RED, ha="left", va="center",
              fontproperties=MONO_BOLD, zorder=7)

    if spec["positions"]:
        axis.text(FIGURE_WIDTH / 2, READOUT_Y - 0.42,
                  " ".join(str(value) for value in OCCURRENCES),
                  fontsize=READOUT_SIZE - 5, color=INK, ha="center", va="center",
                  fontproperties=MONO_BOLD, zorder=7)

    figure.savefig(output_path, transparent=True)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: example_bwt_matching.py OUTPUT.gif")
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
    directory = tempfile.mkdtemp(prefix="bwt_matching_")
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
