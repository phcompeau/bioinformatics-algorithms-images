"""Animated Burrows-Wheeler transform of panamabananas$, in the lecture style.

Reproduces the construction from 02-180 Lecture 5 (Read_Mapping.pptx slides
292-309): form every cyclic rotation of the text, sort them, and read off the
last column. The rotations appear one at a time, then sort into place, then the
last column lifts out as BWT(Text).

The transform is computed from the text and asserted against the published
answer, and additionally checked by inverting it back to the original text.

Run:  python3 example_bwt.py OUTPUT.gif
"""

import math
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

from make_gif import assemble_transparent_gif

BACKGROUND = "#EEE9DF"
INK = "#1A1A1A"
DIM = "#9A958C"
FAINT = "#CFC9BC"
GHOST = "#DCD6C8"
# The deck highlights the live symbol in pure red.
DECK_RED = "#FF0000"
BLUE = "#176FC1"
RED = "#ED1C24"

MONO = font_manager.FontProperties(family="Menlo")
MONO_BOLD = font_manager.FontProperties(family="Menlo", weight="bold")
OPTIMA_ITALIC = font_manager.FontProperties(family="Optima", style="italic")

TEXT = "panamabananas$"
PUBLISHED_BWT = "smnpbnnaaaaa$a"

FIGURE_WIDTH = 12.4
FIGURE_HEIGHT = 6.1
RENDER_DPI = 130
OUTPUT_WIDTH = 1612
OUTPUT_HEIGHT = 793

CHAR_ADVANCE = 0.205
ROW_HEIGHT = 0.305
CHAR_SIZE = 20.0
MATRIX_TOP = 5.30
READOUT_Y = 0.70
READOUT_SIZE = 21.0
LABEL_SIZE = 15.0

# The text as a ring on the right, so "cyclic rotation" is something the viewer
# can see rather than a word: each rotation starts at the letter that lights up.
RING_X = 9.55
RING_Y = 3.05
RING_RADIUS = 1.62
RING_SIZE = 22.0
RING_LABEL_SIZE = 15.0
MATRIX_LEFT_FIXED = 1.25

REVEAL_MS = 460
REVEAL_HOLD_MS = 2200
SORT_FRAMES = 40
SORT_STAGGER = 0.55
SORT_MS = 120
SORT_HOLD_MS = 2600
# The minimal prefixes stay: they show how few letters fix the whole order.
MINIMAL_HOLD_MS = 4600
COLUMN_INTRO_MS = 2200
COLUMN_MS = 420
FINAL_HOLD_MS = 5000


def rotations() -> "list[str]":
    """Every cyclic rotation of the text, starting from offset 0."""
    result = []
    offset = 0
    while offset < len(TEXT):
        result.append(TEXT[offset:] + TEXT[:offset])
        offset = offset + 1
    return result


ROTATIONS = rotations()


def sorted_order() -> "list[int]":
    """Indices of the rotations in lexicographic order."""
    pairs = []
    index = 0
    while index < len(ROTATIONS):
        pairs.append((ROTATIONS[index], index))
        index = index + 1
    pairs.sort()
    order = []
    for rotation, index in pairs:
        order.append(index)
    return order


ORDER = sorted_order()


def last_column() -> str:
    """BWT(Text): the last character of each sorted rotation."""
    letters = []
    for index in ORDER:
        letters.append(ROTATIONS[index][len(TEXT) - 1])
    return "".join(letters)


BWT = last_column()


def first_column() -> str:
    letters = []
    for index in ORDER:
        letters.append(ROTATIONS[index][0])
    return "".join(letters)


FIRST = first_column()


def invert_bwt(transform: str) -> str:
    """Rebuild the text from its transform, to check the transform is right.

    Uses the last-to-first property: the kth occurrence of a symbol in the last
    column is the kth occurrence of that symbol in the first column.
    """
    length = len(transform)
    first_sorted = "".join(sorted(transform))
    seen_last = {}
    last_rank = []
    position = 0
    while position < length:
        symbol = transform[position]
        seen_last[symbol] = seen_last.get(symbol, 0) + 1
        last_rank.append(seen_last[symbol])
        position = position + 1
    first_index = {}
    seen_first = {}
    position = 0
    while position < length:
        symbol = first_sorted[position]
        seen_first[symbol] = seen_first.get(symbol, 0) + 1
        first_index[(symbol, seen_first[symbol])] = position
        position = position + 1
    letters = []
    row = 0
    step = 0
    while step < length:
        symbol = transform[row]
        letters.append(symbol)
        row = first_index[(symbol, last_rank[row])]
        step = step + 1
    letters.reverse()
    recovered = "".join(letters)
    # The walk recovers a cyclic string; the sentinel says where it ends.
    cut = recovered.index("$")
    return recovered[cut + 1:] + recovered[0:cut + 1]


def verify() -> "list[str]":
    """Assert the transform against the published answer and by inversion."""
    lines = []
    size = len(TEXT)

    assert len(ROTATIONS) == size, "one rotation per position"
    assert len(set(ROTATIONS)) == size, "rotations must all be distinct"
    for rotation in ROTATIONS:
        assert len(rotation) == size, "a rotation changed length"
        assert sorted(rotation) == sorted(TEXT), "a rotation is not a rotation"
    lines.append("%d distinct cyclic rotations, each a permutation of the text" % size)

    assert sorted(ORDER) == list(range(size)), "the sort order is not a permutation"
    walker = 1
    while walker < size:
        assert ROTATIONS[ORDER[walker - 1]] < ROTATIONS[ORDER[walker]], (
            "rotations are out of order at row %d" % walker)
        walker = walker + 1
    lines.append("sorted order is a genuine permutation and strictly increasing")

    assert BWT == PUBLISHED_BWT, (
        "computed BWT %r but the slide prints %r" % (BWT, PUBLISHED_BWT))
    lines.append("BWT(Text) = %s, matching the lecture slide exactly" % BWT)

    assert FIRST == "".join(sorted(TEXT)), "first column must be the sorted text"
    lines.append("first column is the text's characters in sorted order")

    assert ROTATIONS[ORDER[0]] == "$" + TEXT[0:size - 1], "row 0 must start with $"
    lines.append("the $ rotation sorts to the top, as the slide shows")

    slot = 1
    while slot < size:
        upper = ROTATIONS[ORDER[slot - 1]]
        lower = ROTATIONS[ORDER[slot]]
        depth = DECIDING[slot]
        assert upper[0:depth - 1] == lower[0:depth - 1], (
            "rows %d and %d were said to agree for %d characters"
            % (slot - 1, slot, depth - 1))
        assert upper[depth - 1] < lower[depth - 1], (
            "rows %d and %d are not decided at character %d" % (slot - 1, slot, depth))
        assert upper[0:MINIMAL[ORDER[slot - 1]]] < lower[0:MINIMAL[ORDER[slot]]], (
            "the minimal prefixes of rows %d and %d do not preserve the order"
            % (slot - 1, slot))
        slot = slot + 1
    total_shown = 0
    deepest = 0
    for row in MINIMAL:
        total_shown = total_shown + MINIMAL[row]
        if MINIMAL[row] > deepest:
            deepest = MINIMAL[row]
    lines.append("the sort is settled by %d of the %d characters in the block, "
                 "at most %d in any row" % (total_shown, size * size, deepest))

    recovered = invert_bwt(BWT)
    assert recovered == TEXT, (
        "inverting the transform gave %r, not the original text" % recovered)
    lines.append("inverting BWT(Text) reproduces %s exactly" % TEXT)

    assert (MATRIX_LEFT_FIXED + size * CHAR_ADVANCE + 0.4
            < RING_X - RING_RADIUS - 0.3), "the matrix collides with the ring"
    assert RING_X + RING_RADIUS + 0.3 < FIGURE_WIDTH, "the ring runs off the right"
    assert RING_Y + RING_RADIUS + 0.3 < FIGURE_HEIGHT - 0.5, "the ring hits the caption"
    assert RING_Y - RING_RADIUS - 0.62 > READOUT_Y, "the ring hits the readout"
    matrix_width = size * CHAR_ADVANCE
    lowest = MATRIX_TOP - (size - 1) * ROW_HEIGHT
    assert matrix_width < FIGURE_WIDTH - 1.0, "matrix too wide for the canvas"
    assert lowest > READOUT_Y + 0.45, "matrix collides with the readout"
    assert MATRIX_TOP + 0.3 < FIGURE_HEIGHT, "matrix runs off the top"
    lines.append("matrix is %.2f x %.2f in, clear of the caption and readout"
                 % (matrix_width, (size - 1) * ROW_HEIGHT))
    return lines


def common_prefix_length(first: str, second: str) -> int:
    """How far two strings agree before they first differ."""
    shared = 0
    while shared < len(first) and shared < len(second):
        if first[shared] != second[shared]:
            return shared
        shared = shared + 1
    return shared


def adjacent_prefixes() -> "list[int]":
    """How many characters settle each neighbouring pair in the sorted block."""
    result = [0]
    slot = 1
    while slot < len(ORDER):
        upper = ROTATIONS[ORDER[slot - 1]]
        lower = ROTATIONS[ORDER[slot]]
        result.append(common_prefix_length(upper, lower) + 1)
        slot = slot + 1
    return result


DECIDING = adjacent_prefixes()


def minimal_prefixes() -> dict:
    """row -> how many characters that row needs to hold its place in the sort."""
    result = {}
    slot = 0
    while slot < len(ORDER):
        needed = 1
        if slot > 0 and DECIDING[slot] > needed:
            needed = DECIDING[slot]
        if slot + 1 < len(ORDER) and DECIDING[slot + 1] > needed:
            needed = DECIDING[slot + 1]
        result[ORDER[slot]] = needed
        slot = slot + 1
    return result


MINIMAL = minimal_prefixes()


def matrix_left() -> float:
    """Fixed, because the ring now takes the right-hand side of the frame."""
    return MATRIX_LEFT_FIXED


LEFT = matrix_left()


def row_y(slot: float) -> float:
    return MATRIX_TOP - slot * ROW_HEIGHT


def char_x(column: int) -> float:
    return LEFT + (column + 0.5) * CHAR_ADVANCE


def ease(fraction: float) -> float:
    clamped = min(max(fraction, 0.0), 1.0)
    return 0.5 * (1 - math.cos(math.pi * clamped))


def new_axes() -> "tuple":
    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=RENDER_DPI)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0, FIGURE_WIDTH)
    axis.set_ylim(0, FIGURE_HEIGHT)
    axis.set_aspect("equal")
    axis.axis("off")
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    return (figure, axis)


def base_frame(duration: int) -> dict:
    return {"revealed": 0, "slots": {}, "alphas": {}, "highlight_columns": 0,
            "dim_rest": False, "caption": "", "readout": 0, "prefix_limit": {},
            "mark": {}, "focus": None, "ring_start": -1, "duration_ms": duration}


def build_specs() -> "list[dict]":
    """Reveal the rotations, sort them, then lift out the last column."""
    specs = []
    size = len(TEXT)

    unsorted_slots = {}
    row = 0
    while row < size:
        unsorted_slots[row] = float(row)
        row = row + 1

    sorted_slots = {}
    slot = 0
    while slot < size:
        sorted_slots[ORDER[slot]] = float(slot)
        slot = slot + 1

    reveal = 1
    while reveal <= size:
        frame = base_frame(REVEAL_MS)
        frame["revealed"] = reveal
        frame["slots"] = dict(unsorted_slots)
        frame["ring_start"] = reveal - 1
        frame["caption"] = ("Each rotation reads the ring from one letter onward")
        specs.append(frame)
        reveal = reveal + 1

    hold = base_frame(REVEAL_HOLD_MS)
    hold["revealed"] = size
    hold["slots"] = dict(unsorted_slots)
    hold["caption"] = "All %d cyclic rotations of the text" % size
    specs.append(hold)

    # Sorting: each row slides to its sorted slot, staggered so the rows leave
    # in a wave rather than crossing all at once.
    step = 1
    while step <= SORT_FRAMES:
        frame = base_frame(SORT_MS)
        frame["revealed"] = size
        frame["caption"] = "Sort the rotations"
        slots = {}
        alphas = {}
        row = 0
        while row < size:
            delay = SORT_STAGGER * (row / (size - 1.0))
            local = (step / SORT_FRAMES - delay) / (1.0 - SORT_STAGGER)
            clamped = min(max(local, 0.0), 1.0)
            eased = ease(clamped)
            slots[row] = (unsorted_slots[row]
                          + (sorted_slots[row] - unsorted_slots[row]) * eased)
            # Rows cross each other, so fade one while it is in flight; at full
            # opacity they pile into an illegible smear in the middle.
            if unsorted_slots[row] == sorted_slots[row]:
                alphas[row] = 1.0
            else:
                alphas[row] = 1.0 - 0.78 * math.sin(math.pi * clamped)
            row = row + 1
        frame["slots"] = slots
        frame["alphas"] = alphas
        specs.append(frame)
        step = step + 1

    settled = base_frame(SORT_HOLD_MS)
    settled["revealed"] = size
    settled["slots"] = dict(sorted_slots)
    settled["caption"] = "Sort the rotations"
    specs.append(settled)

    minimal = base_frame(MINIMAL_HOLD_MS)
    minimal["revealed"] = size
    minimal["slots"] = dict(sorted_slots)
    minimal["prefix_limit"] = dict(MINIMAL)
    minimal["caption"] = "Only these letters matter to the sorted order"
    specs.append(minimal)

    hand_off = base_frame(COLUMN_INTRO_MS)
    hand_off["revealed"] = size
    hand_off["slots"] = dict(sorted_slots)
    hand_off["dim_rest"] = True
    hand_off["caption"] = "BWT(Text) is the last column"
    specs.append(hand_off)

    lift = 1
    while lift <= size:
        frame = base_frame(COLUMN_MS)
        frame["revealed"] = size
        frame["slots"] = dict(sorted_slots)
        frame["highlight_columns"] = lift
        frame["dim_rest"] = True
        frame["readout"] = lift
        frame["caption"] = "BWT(Text) is the last column"
        specs.append(frame)
        lift = lift + 1

    final = base_frame(FINAL_HOLD_MS)
    final["revealed"] = size
    final["slots"] = dict(sorted_slots)
    final["highlight_columns"] = size
    final["dim_rest"] = True
    final["readout"] = size
    final["caption"] = "BWT(Text) is the last column"
    specs.append(final)
    return specs


def draw_ring(axis: "plt.Axes", start: int) -> None:
    """The text written round a circle, which is what makes it cyclic.

    Position 0 sits at the top and the letters run clockwise, so a rotation is
    just a starting point on the ring. The letter a rotation begins at is drawn in
    the deck's red, with an arrow from the middle pointing at it.
    """
    import matplotlib.patches as patches
    guide = patches.Circle((RING_X, RING_Y), RING_RADIUS, facecolor="none",
                           edgecolor=FAINT, linewidth=1.0, zorder=2)
    axis.add_patch(guide)
    position = 0
    while position < len(TEXT):
        angle = math.pi / 2 - 2 * math.pi * position / len(TEXT)
        x = RING_X + RING_RADIUS * math.cos(angle)
        y = RING_Y + RING_RADIUS * math.sin(angle)
        if position == start:
            colour = DECK_RED
            font = MONO_BOLD
        else:
            colour = INK
            font = MONO
        axis.text(x, y, TEXT[position], fontsize=RING_SIZE, color=colour,
                  ha="center", va="center", fontproperties=font, zorder=5)
        position = position + 1
    if start >= 0:
        angle = math.pi / 2 - 2 * math.pi * start / len(TEXT)
        near = RING_RADIUS * 0.52
        far = RING_RADIUS - 0.30
        axis.annotate("", xy=(RING_X + far * math.cos(angle),
                              RING_Y + far * math.sin(angle)),
                      xytext=(RING_X + near * math.cos(angle),
                              RING_Y + near * math.sin(angle)),
                      arrowprops={"arrowstyle": "-|>", "color": DECK_RED,
                                  "linewidth": 1.6, "shrinkA": 0, "shrinkB": 0},
                      zorder=4)
        axis.text(RING_X, RING_Y - RING_RADIUS - 0.42, "start at %d" % start,
                  fontsize=RING_LABEL_SIZE, color=DECK_RED, ha="center",
                  va="center", fontproperties=MONO, zorder=5)


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes()
    size = len(TEXT)
    last = size - 1

    row = 0
    while row < size:
        if row >= spec["revealed"]:
            row = row + 1
            continue
        y = row_y(spec["slots"][row])
        rotation = ROTATIONS[row]
        column = 0
        while column < size:
            highlighted = False
            if spec["highlight_columns"] > 0 and column == last:
                slot = int(round(spec["slots"][row]))
                if slot < spec["highlight_columns"]:
                    highlighted = True
            dulled = spec["focus"] is not None and row not in spec["focus"]
            limit = spec["prefix_limit"].get(row, -1)
            if highlighted:
                colour = INK
                font = MONO_BOLD
            elif dulled:
                colour = GHOST
                font = MONO
            elif column == spec["mark"].get(row, -1):
                colour = BLUE
                font = MONO
            elif limit >= 0 and column >= limit:
                colour = FAINT
                font = MONO
            elif spec["dim_rest"]:
                colour = DIM
                font = MONO
            else:
                colour = INK
                font = MONO
            axis.text(char_x(column), y, rotation[column], fontsize=CHAR_SIZE,
                      color=colour, ha="center", va="center", fontproperties=font,
                      zorder=5, alpha=spec["alphas"].get(row, 1.0))
            column = column + 1
        row = row + 1

    draw_ring(axis, spec["ring_start"])

    if spec["caption"] != "":
        axis.text(FIGURE_WIDTH / 2, FIGURE_HEIGHT - 0.42, spec["caption"],
                  fontsize=LABEL_SIZE, color=INK, ha="center", va="center",
                  fontproperties=OPTIMA_ITALIC, zorder=6)

    if spec["readout"] > 0:
        shown = BWT[0:spec["readout"]]
        axis.text(FIGURE_WIDTH / 2, READOUT_Y, shown, fontsize=READOUT_SIZE,
                  color=INK, ha="center", va="center", fontproperties=MONO_BOLD,
                  zorder=6)

    figure.savefig(output_path, transparent=True)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: example_bwt.py OUTPUT.gif")
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
    directory = tempfile.mkdtemp(prefix="bwt_")
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
