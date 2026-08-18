"""Animated suffix array construction, in the 02-180 lecture style.

Follows Read_Mapping.pptx slides 251 onward: list every suffix of the text with
the position it starts at, sort the suffixes lexicographically, and the column
of starting positions left behind is the suffix array.

The suffix array is computed from the text and asserted against the leaf order
of the suffix tree, which was itself verified against the published figure.

Run:  python3 example_suffix_array.py OUTPUT.gif
"""

import os
import sys
import tempfile

from lecture_style import (BACKGROUND, BLUE, DIM, DISC, FAINT, INK, MONO,
                           MONO_BOLD, OPTIMA_ITALIC, ease, new_axes,
                           sort_permutation, staggered_progress, transit_alpha)
from make_gif import assemble_transparent_gif

TEXT = "panamabananas$"

# Read left to right off the leaves of images/BWT/suffix_tree.png, which is the
# same order the suffix array must be in.
PUBLISHED_ARRAY = [13, 5, 3, 1, 7, 9, 11, 6, 4, 2, 8, 10, 0, 12]

FIGURE_WIDTH = 9.4
FIGURE_HEIGHT = 7.0
RENDER_DPI = 140
OUTPUT_WIDTH = 1316
OUTPUT_HEIGHT = 980

CHAR_ADVANCE = 0.205
ROW_HEIGHT = 0.305
CHAR_SIZE = 20.0
MATRIX_TOP = 6.10
INDEX_GAP = 0.62
DISC_RADIUS = 0.145
INDEX_SIZE = 12.0
LABEL_SIZE = 15.0
CAPTION_Y = 6.60
# Three lines under the block: the text itself, a ruler of positions beneath it,
# and the suffix array being read off. The ruler is what turns the array from a
# list of numbers into a set of places inside the text.
GHOST = "#DCD6C8"
TEXT_ADVANCE = 0.30
TEXT_Y = 1.44
RULER_Y = 1.06
READOUT_Y = 0.50
READOUT_SIZE = 17.0
RULER_SIZE = 11.0

REVEAL_MS = 330
REVEAL_HOLD_MS = 2200
SORT_FRAMES = 40
SORT_STAGGER = 0.55
SORT_MS = 120
SORT_HOLD_MS = 2400
# The proof that the sort is right: each adjacent pair only has to be compared as
# far as the first place they differ.
PAIR_MS = 900
PAIR_INTRO_MS = 2600
MINIMAL_HOLD_MS = 4200
# Phillip: the hand-off to "these are the suffix array" went by too fast.
PICK_INTRO_MS = 2400
PICK_MS = 700
FINAL_HOLD_MS = 5600


def suffixes() -> "list[str]":
    result = []
    start = 0
    while start < len(TEXT):
        result.append(TEXT[start:])
        start = start + 1
    return result


SUFFIXES = suffixes()
ORDER = sort_permutation(SUFFIXES)


def suffix_array() -> "list[int]":
    """Starting positions of the suffixes in sorted order."""
    result = []
    for index in ORDER:
        result.append(index)
    return result


ARRAY = suffix_array()


def common_prefix_length(first: str, second: str) -> int:
    """How far two strings agree before they first differ."""
    shared = 0
    while shared < len(first) and shared < len(second):
        if first[shared] != second[shared]:
            return shared
        shared = shared + 1
    return shared


def adjacent_prefixes() -> "list[int]":
    """For each sorted neighbour pair, how many characters settle the comparison.

    The pair is decided at the first position where the two suffixes differ, so
    everything past that position is irrelevant to the sort. This is what the
    animation shows: the order is already proved by a handful of letters.
    """
    result = [0]
    slot = 1
    while slot < len(ARRAY):
        upper = SUFFIXES[ARRAY[slot - 1]]
        lower = SUFFIXES[ARRAY[slot]]
        result.append(common_prefix_length(upper, lower) + 1)
        slot = slot + 1
    return result


DECIDING = adjacent_prefixes()


def minimal_prefixes() -> dict:
    """rows -> how many characters that row needs to hold its place in the sort.

    A row is pinned by both of its neighbours, so it needs whichever comparison
    reaches deeper.
    """
    result = {}
    slot = 0
    while slot < len(ARRAY):
        needed = 1
        if slot > 0 and DECIDING[slot] > needed:
            needed = DECIDING[slot]
        if slot + 1 < len(ARRAY) and DECIDING[slot + 1] > needed:
            needed = DECIDING[slot + 1]
        result[ARRAY[slot]] = needed
        slot = slot + 1
    return result


MINIMAL = minimal_prefixes()


def verify() -> "list[str]":
    lines = []
    size = len(TEXT)

    assert len(SUFFIXES) == size, "one suffix per position"
    assert len(set(SUFFIXES)) == size, "suffixes must be distinct"
    index = 0
    while index < size:
        assert SUFFIXES[index] == TEXT[index:], "suffix %d is wrong" % index
        index = index + 1
    lines.append("%d distinct suffixes, each the tail of the text" % size)

    walker = 1
    while walker < size:
        assert SUFFIXES[ORDER[walker - 1]] < SUFFIXES[ORDER[walker]], (
            "sorted suffixes out of order at row %d" % walker)
        walker = walker + 1
    assert sorted(ARRAY) == list(range(size)), "the array is not a permutation"
    lines.append("sorted order is a permutation and strictly increasing")

    assert ARRAY == PUBLISHED_ARRAY, (
        "computed suffix array %s but the suffix tree leaves give %s"
        % (ARRAY, PUBLISHED_ARRAY))
    lines.append("suffix array is %s, matching the suffix tree's leaf order"
                 % " ".join(str(value) for value in ARRAY))

    assert ARRAY[0] == size - 1, "the sentinel suffix must sort first"
    lines.append("the lone $ suffix sorts to the top")

    # The minimal prefixes must really be enough: truncating every suffix to its
    # own minimal prefix has to leave the rows in the same order.
    slot = 1
    while slot < size:
        upper = SUFFIXES[ARRAY[slot - 1]]
        lower = SUFFIXES[ARRAY[slot]]
        depth = DECIDING[slot]
        assert upper[0:depth - 1] == lower[0:depth - 1], (
            "rows %d and %d were said to agree for %d characters" % (slot - 1, slot, depth - 1))
        assert upper[depth - 1] < lower[depth - 1], (
            "rows %d and %d are not decided at character %d" % (slot - 1, slot, depth))
        assert upper[0:MINIMAL[ARRAY[slot - 1]]] < lower[0:MINIMAL[ARRAY[slot]]], (
            "the minimal prefixes of rows %d and %d do not preserve the order"
            % (slot - 1, slot))
        slot = slot + 1
    deepest = 0
    for row in MINIMAL:
        if MINIMAL[row] > deepest:
            deepest = MINIMAL[row]
    total_shown = 0
    for row in MINIMAL:
        total_shown = total_shown + MINIMAL[row]
    lines.append("the sort is settled by %d of the %d characters on screen, "
                 "at most %d in any row"
                 % (total_shown, size * (size + 1) // 2, deepest))

    widest = 0
    for suffix in SUFFIXES:
        if len(suffix) > widest:
            widest = len(suffix)
    total = INDEX_GAP + widest * CHAR_ADVANCE
    assert total < FIGURE_WIDTH - 1.0, "matrix too wide for the canvas"
    lowest = MATRIX_TOP - (size - 1) * ROW_HEIGHT
    assert lowest - DISC_RADIUS > TEXT_Y + 0.30, "matrix collides with the text below it"
    assert TEXT_Y > RULER_Y + 0.25, "the ruler collides with the text"
    assert RULER_Y > READOUT_Y + 0.30, "the ruler collides with the suffix array"
    assert MATRIX_TOP + 0.25 < CAPTION_Y, "matrix collides with the caption"
    lines.append("block is %.2f x %.2f in, clear of the caption, the text and the array"
                 % (total, (size - 1) * ROW_HEIGHT))
    return lines


def block_left() -> float:
    widest = 0
    for suffix in SUFFIXES:
        if len(suffix) > widest:
            widest = len(suffix)
    return (FIGURE_WIDTH - (INDEX_GAP + widest * CHAR_ADVANCE)) / 2


LEFT = block_left()


def row_y(slot: float) -> float:
    return MATRIX_TOP - slot * ROW_HEIGHT


def base_frame(duration: int) -> dict:
    return {"revealed": 0, "slots": {}, "alphas": {}, "picked": 0,
            "dim_text": False, "caption": "", "prefix_limit": {}, "mark": {},
            "focus": None, "duration_ms": duration}


def build_specs() -> "list[dict]":
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
        frame["caption"] = "Every suffix of %s, with where it starts" % TEXT
        specs.append(frame)
        reveal = reveal + 1

    hold = base_frame(REVEAL_HOLD_MS)
    hold["revealed"] = size
    hold["slots"] = dict(unsorted_slots)
    hold["caption"] = "Every suffix of %s, with where it starts" % TEXT
    specs.append(hold)

    step = 1
    while step <= SORT_FRAMES:
        frame = base_frame(SORT_MS)
        frame["revealed"] = size
        frame["caption"] = "Sort the suffixes"
        slots = {}
        alphas = {}
        row = 0
        while row < size:
            progress = staggered_progress(row, size, step / SORT_FRAMES, SORT_STAGGER)
            slots[row] = (unsorted_slots[row]
                          + (sorted_slots[row] - unsorted_slots[row]) * progress)
            alphas[row] = transit_alpha(progress,
                                        unsorted_slots[row] != sorted_slots[row])
            row = row + 1
        frame["slots"] = slots
        frame["alphas"] = alphas
        specs.append(frame)
        step = step + 1

    settled = base_frame(SORT_HOLD_MS)
    settled["revealed"] = size
    settled["slots"] = dict(sorted_slots)
    settled["caption"] = "Sort the suffixes"
    specs.append(settled)

    # Why the order is right: walk the neighbours and show that each pair is
    # already decided at the first letter where the two suffixes differ.
    slot = 1
    while slot < size:
        upper = ARRAY[slot - 1]
        lower = ARRAY[slot]
        frame = base_frame(PAIR_MS)
        frame["revealed"] = size
        frame["slots"] = dict(sorted_slots)
        frame["focus"] = set([upper, lower])
        frame["prefix_limit"] = {upper: DECIDING[slot], lower: DECIDING[slot]}
        frame["mark"] = {upper: DECIDING[slot] - 1, lower: DECIDING[slot] - 1}
        frame["caption"] = "Each pair is settled at the first letter where they differ"
        specs.append(frame)
        slot = slot + 1

    minimal = base_frame(MINIMAL_HOLD_MS)
    minimal["revealed"] = size
    minimal["slots"] = dict(sorted_slots)
    minimal["prefix_limit"] = dict(MINIMAL)
    minimal["caption"] = "These few letters already prove the whole order"
    specs.append(minimal)

    hand_off = base_frame(PICK_INTRO_MS)
    hand_off["revealed"] = size
    hand_off["slots"] = dict(sorted_slots)
    hand_off["dim_text"] = True
    hand_off["caption"] = "The starting positions are the suffix array"
    specs.append(hand_off)

    pick = 1
    while pick <= size:
        frame = base_frame(PICK_MS)
        frame["revealed"] = size
        frame["slots"] = dict(sorted_slots)
        frame["picked"] = pick
        frame["dim_text"] = True
        frame["caption"] = "The starting positions are the suffix array"
        specs.append(frame)
        pick = pick + 1

    final = base_frame(FINAL_HOLD_MS)
    final["revealed"] = size
    final["slots"] = dict(sorted_slots)
    final["picked"] = size
    final["dim_text"] = True
    final["caption"] = "The starting positions are the suffix array"
    specs.append(final)
    return specs


def draw_frame(spec: dict, output_path: str) -> None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)
    size = len(TEXT)

    row = 0
    while row < size:
        if row >= spec["revealed"]:
            row = row + 1
            continue
        y = row_y(spec["slots"][row])
        alpha = spec["alphas"].get(row, 1.0)
        slot = int(round(spec["slots"][row]))
        highlighted = spec["picked"] > 0 and slot < spec["picked"]

        if highlighted:
            disc = patches.Circle((LEFT + DISC_RADIUS, y), DISC_RADIUS,
                                  facecolor=DISC, edgecolor="none", zorder=5,
                                  alpha=alpha)
            axis.add_patch(disc)
            index_colour = "white"
        else:
            index_colour = INK
        axis.text(LEFT + DISC_RADIUS, y, str(row), fontsize=INDEX_SIZE,
                  color=index_colour, ha="center", va="center",
                  fontproperties=MONO_BOLD, zorder=6, alpha=alpha)

        if spec["dim_text"]:
            text_colour = DIM
        else:
            text_colour = INK
        dulled = spec["focus"] is not None and row not in spec["focus"]
        limit = spec["prefix_limit"].get(row, -1)
        marked = spec["mark"].get(row, -1)
        suffix = SUFFIXES[row]
        column = 0
        while column < len(suffix):
            colour = text_colour
            if dulled:
                colour = GHOST
            elif limit >= 0 and column >= limit:
                colour = FAINT
            if column == marked and not dulled:
                colour = BLUE
            axis.text(LEFT + INDEX_GAP + (column + 0.5) * CHAR_ADVANCE, y,
                      suffix[column], fontsize=CHAR_SIZE, color=colour,
                      ha="center", va="center", fontproperties=MONO, zorder=5,
                      alpha=alpha)
            column = column + 1
        row = row + 1

    if spec["caption"] != "":
        axis.text(FIGURE_WIDTH / 2, CAPTION_Y, spec["caption"], fontsize=LABEL_SIZE,
                  color=INK, ha="center", va="center", fontproperties=OPTIMA_ITALIC,
                  zorder=7)

    # The text with a ruler of positions under it, so the numbers being read off
    # the block are visibly places inside panamabananas$ rather than bare digits.
    taken = set()
    index = 0
    while index < spec["picked"]:
        taken.add(ARRAY[index])
        index = index + 1
    current = -1
    if spec["picked"] > 0:
        current = ARRAY[spec["picked"] - 1]
    text_left = (FIGURE_WIDTH - len(TEXT) * TEXT_ADVANCE) / 2
    position = 0
    while position < len(TEXT):
        x = text_left + (position + 0.5) * TEXT_ADVANCE
        if position == current:
            letter_colour = BLUE
            ruler_colour = BLUE
        elif position in taken:
            letter_colour = INK
            ruler_colour = DISC
        else:
            letter_colour = DIM
            ruler_colour = FAINT
        axis.text(x, TEXT_Y, TEXT[position], fontsize=CHAR_SIZE, color=letter_colour,
                  ha="center", va="center", fontproperties=MONO_BOLD, zorder=7)
        axis.text(x, RULER_Y, str(position), fontsize=RULER_SIZE, color=ruler_colour,
                  ha="center", va="center", fontproperties=MONO, zorder=7)
        position = position + 1

    if spec["picked"] > 0:
        shown = []
        index = 0
        while index < spec["picked"]:
            shown.append(str(ARRAY[index]))
            index = index + 1
        axis.text(FIGURE_WIDTH / 2, READOUT_Y, " ".join(shown), fontsize=READOUT_SIZE,
                  color=INK, ha="center", va="center", fontproperties=MONO_BOLD,
                  zorder=7)

    figure.savefig(output_path, transparent=True)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: example_suffix_array.py OUTPUT.gif")
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
    directory = tempfile.mkdtemp(prefix="suffix_array_")
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
