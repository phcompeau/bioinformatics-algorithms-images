"""Animated inversion of the Burrows-Wheeler transform, in the lecture style.

Follows Read_Mapping.pptx slides 318-340: knowing only BWT(Text), rebuild the
sorted rotation matrix one column at a time. Prepending the last column to the
known prefix of each row gives every (k+1)-mer of the cyclic text; sorting those
gives the first k+1 columns. Repeat until the matrix is complete, and the row
that ends in the sentinel is the text.

Every stage is asserted against the true sorted rotation matrix, so a wrong
column cannot survive to a frame.

Run:  python3 bwt_inversion.py OUTPUT.gif
"""

import os
import sys
import tempfile

import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, DIM, FAINT, INK, MONO, MONO_BOLD, RED,
                           OPTIMA_ITALIC, ease, new_axes, staggered_progress,
                           transit_alpha)
from make_gif import assemble_gif

TEXT = "banana$"
PUBLISHED_BWT = "annb$aa"

FIGURE_WIDTH = 8.6
FIGURE_HEIGHT = 5.2
RENDER_DPI = 150
OUTPUT_WIDTH = 1290
OUTPUT_HEIGHT = 780

CHAR_ADVANCE = 0.30
ROW_HEIGHT = 0.42
CHAR_SIZE = 24.0
MATRIX_TOP = 4.05
CAPTION_Y = 4.72
READOUT_Y = 0.52
READOUT_SIZE = 22.0
LABEL_SIZE = 15.0

PREPEND_FRAMES = 9
PREPEND_MS = 110
SORT_FRAMES = 13
SORT_MS = 110
SORT_STAGGER = 0.5
STAGE_HOLD_MS = 1500
FIRST_HOLD_MS = 2600
FINAL_HOLD_MS = 5200


def rotations() -> "list[str]":
    result = []
    offset = 0
    while offset < len(TEXT):
        result.append(TEXT[offset:] + TEXT[:offset])
        offset = offset + 1
    return result


TRUE_MATRIX = sorted(rotations())


def bwt() -> str:
    letters = []
    for row in TRUE_MATRIX:
        letters.append(row[len(TEXT) - 1])
    return "".join(letters)


BWT = bwt()


def stages() -> "list[dict]":
    """Rebuild the matrix column by column, recording each stage.

    Each stage holds the prefixes before prepending, the (k+1)-mers formed by
    prepending the last column, and the sorted result. Ties are broken by the
    previous row order, which is the refinement the argument relies on.
    """
    size = len(TEXT)
    prefixes = []
    row = 0
    while row < size:
        prefixes.append("")
        row = row + 1
    record = []
    width = 0
    while width < size:
        formed = []
        row = 0
        while row < size:
            formed.append(BWT[row] + prefixes[row])
            row = row + 1
        pairs = []
        row = 0
        while row < size:
            pairs.append((formed[row], row))
            row = row + 1
        pairs.sort()
        order = []
        sorted_prefixes = []
        for value, row in pairs:
            order.append(row)
            sorted_prefixes.append(value)
        record.append({"before": list(prefixes), "formed": list(formed),
                       "order": list(order), "after": list(sorted_prefixes),
                       "width": width + 1})
        prefixes = sorted_prefixes
        width = width + 1
    return record


STAGES = stages()


def verify() -> "list[str]":
    lines = []
    size = len(TEXT)

    assert len(TRUE_MATRIX) == size, "one sorted rotation per position"
    assert BWT == PUBLISHED_BWT, (
        "computed BWT %r but the slide prints %r" % (BWT, PUBLISHED_BWT))
    lines.append("BWT(%s) = %s, matching the lecture slide" % (TEXT, BWT))

    assert len(STAGES) == size, "one stage per column"
    for stage in STAGES:
        width = stage["width"]
        expected = []
        for row in TRUE_MATRIX:
            expected.append(row[0:width])
        assert stage["after"] == expected, (
            "stage %d rebuilt %s but the matrix has %s"
            % (width, stage["after"], expected))
        assert sorted(stage["order"]) == list(range(size)), (
            "stage %d order is not a permutation" % width)
        walker = 1
        while walker < size:
            assert stage["after"][walker - 1] <= stage["after"][walker], (
                "stage %d is not sorted" % width)
            walker = walker + 1
    lines.append("all %d rebuilt columns match the true sorted rotation matrix"
                 % size)

    for stage in STAGES:
        row = 0
        while row < size:
            assert stage["formed"][row] == BWT[row] + stage["before"][row], (
                "a formed k-mer is not last-column plus prefix")
            row = row + 1
        assert sorted(stage["formed"]) == sorted(stage["after"]), (
            "sorting the formed k-mers must give the stage's columns")
    lines.append("every stage forms its k-mers as last column plus known prefix")

    final = STAGES[len(STAGES) - 1]["after"]
    assert final == TRUE_MATRIX, "the last stage must be the whole matrix"
    recovered = ""
    for row in final:
        if row[len(TEXT) - 1] == "$":
            recovered = row
    assert recovered == TEXT, (
        "the row ending in $ is %r, not the text" % recovered)
    lines.append("the completed matrix's $-terminated row is %s, the original text"
                 % recovered)

    matrix_width = size * CHAR_ADVANCE
    lowest = MATRIX_TOP - (size - 1) * ROW_HEIGHT
    assert matrix_width < FIGURE_WIDTH - 1.0, "matrix too wide"
    assert lowest > READOUT_Y + 0.45, "matrix collides with the readout"
    assert MATRIX_TOP + 0.28 < CAPTION_Y, "matrix collides with the caption"
    lines.append("matrix is %.2f x %.2f in, clear of the caption and readout"
                 % (matrix_width, (size - 1) * ROW_HEIGHT))
    return lines


LEFT = (FIGURE_WIDTH - len(TEXT) * CHAR_ADVANCE) / 2


def char_x(column: float) -> float:
    return LEFT + (column + 0.5) * CHAR_ADVANCE


def row_y(slot: float) -> float:
    return MATRIX_TOP - slot * ROW_HEIGHT


def base_frame(duration: int) -> dict:
    return {"rows": [], "caption": "", "readout": False, "duration_ms": duration}


def row_state(text: str, slot: float, alpha: float, shift: float,
              flying: str, fly_from: float, fly_to: float) -> dict:
    """One row: its known prefix, where it sits, and any character in flight."""
    return {"text": text, "slot": slot, "alpha": alpha, "shift": shift,
            "flying": flying, "fly_from": fly_from, "fly_to": fly_to}


def build_specs() -> "list[dict]":
    specs = []
    size = len(TEXT)

    opening = base_frame(FIRST_HOLD_MS)
    rows = []
    row = 0
    while row < size:
        rows.append(row_state("", float(row), 1.0, 0.0, "", 0.0, 0.0))
        row = row + 1
    opening["rows"] = rows
    opening["caption"] = "All we know is the last column, BWT(Text) = %s" % BWT
    specs.append(opening)

    stage_index = 0
    while stage_index < len(STAGES):
        stage = STAGES[stage_index]
        before = stage["before"]
        formed = stage["formed"]
        order = stage["order"]
        after = stage["after"]

        # Prepend: the last column's character travels to the front of its row
        # while the known prefix shifts one column to the right.
        step = 1
        while step <= PREPEND_FRAMES:
            fraction = ease(step / PREPEND_FRAMES)
            frame = base_frame(PREPEND_MS)
            rows = []
            row = 0
            while row < size:
                rows.append(row_state(before[row], float(row), 1.0, fraction,
                                      BWT[row], float(size - 1),
                                      0.0))
                rows[row]["fly_progress"] = fraction
                row = row + 1
            frame["rows"] = rows
            frame["caption"] = "Prepend the last column to what we know"
            specs.append(frame)
            step = step + 1

        formed_hold = base_frame(STAGE_HOLD_MS)
        rows = []
        row = 0
        while row < size:
            rows.append(row_state(formed[row], float(row), 1.0, 0.0, "", 0.0, 0.0))
            row = row + 1
        formed_hold["rows"] = rows
        formed_hold["caption"] = ("Every %d-mer of the text, but out of order"
                                 % stage["width"])
        specs.append(formed_hold)

        target_slot = {}
        slot = 0
        while slot < size:
            target_slot[order[slot]] = float(slot)
            slot = slot + 1

        step = 1
        while step <= SORT_FRAMES:
            frame = base_frame(SORT_MS)
            rows = []
            row = 0
            while row < size:
                progress = staggered_progress(row, size, step / SORT_FRAMES,
                                              SORT_STAGGER)
                slot_now = float(row) + (target_slot[row] - float(row)) * progress
                moving = target_slot[row] != float(row)
                rows.append(row_state(formed[row], slot_now,
                                      transit_alpha(progress, moving), 0.0,
                                      "", 0.0, 0.0))
                row = row + 1
            frame["rows"] = rows
            frame["caption"] = "Sort them to get the first %d columns" % stage["width"]
            specs.append(frame)
            step = step + 1

        settled = base_frame(STAGE_HOLD_MS)
        rows = []
        row = 0
        while row < size:
            rows.append(row_state(after[row], float(row), 1.0, 0.0, "", 0.0, 0.0))
            row = row + 1
        settled["rows"] = rows
        if stage["width"] < size:
            settled["caption"] = ("Sort them to get the first %d columns"
                                  % stage["width"])
        else:
            settled["caption"] = "The matrix is complete"
        specs.append(settled)
        stage_index = stage_index + 1

    final = base_frame(FINAL_HOLD_MS)
    rows = []
    row = 0
    while row < size:
        rows.append(row_state(TRUE_MATRIX[row], float(row), 1.0, 0.0, "", 0.0, 0.0))
        row = row + 1
    final["rows"] = rows
    final["caption"] = "The row that ends in $ is the text"
    final["readout"] = True
    specs.append(final)
    return specs


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)
    size = len(TEXT)

    row = 0
    while row < len(spec["rows"]):
        state = spec["rows"][row]
        y = row_y(state["slot"])
        alpha = state["alpha"]
        known = state["text"]

        # faint placeholders for the columns still unknown
        column = len(known)
        while column < size - 1:
            # While the prefix shifts right, drop any placeholder that would slide
            # under the last column.
            if column + state["shift"] <= size - 2 + 1e-9:
                axis.text(char_x(column + state["shift"]), y, ".",
                          fontsize=CHAR_SIZE, color=FAINT, ha="center",
                          va="center", fontproperties=MONO, zorder=4, alpha=alpha)
            column = column + 1

        column = 0
        while column < len(known):
            highlight = INK
            if spec["readout"] and known[size - 1] == "$":
                highlight = RED
            axis.text(char_x(column + state["shift"]), y, known[column],
                      fontsize=CHAR_SIZE, color=highlight, ha="center",
                      va="center", fontproperties=MONO, zorder=5, alpha=alpha)
            column = column + 1

        if len(known) < size:
            axis.text(char_x(size - 1), y, BWT[row], fontsize=CHAR_SIZE,
                      color=INK, ha="center", va="center",
                      fontproperties=MONO_BOLD, zorder=5, alpha=alpha)

        if state["flying"] != "":
            progress = state.get("fly_progress", 0.0)
            start = float(size - 1)
            end = 0.0
            axis.text(char_x(start + (end - start) * progress), y, state["flying"],
                      fontsize=CHAR_SIZE, color=RED, ha="center", va="center",
                      fontproperties=MONO_BOLD, zorder=6, alpha=alpha)
        row = row + 1

    if spec["caption"] != "":
        axis.text(FIGURE_WIDTH / 2, CAPTION_Y, spec["caption"], fontsize=LABEL_SIZE,
                  color=INK, ha="center", va="center", fontproperties=OPTIMA_ITALIC,
                  zorder=7)

    if spec["readout"]:
        axis.text(FIGURE_WIDTH / 2, READOUT_Y, TEXT, fontsize=READOUT_SIZE,
                  color=RED, ha="center", va="center", fontproperties=MONO_BOLD,
                  zorder=7)

    figure.savefig(output_path, facecolor=BACKGROUND)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: bwt_inversion.py OUTPUT.gif")
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
    directory = tempfile.mkdtemp(prefix="bwt_inversion_")
    frame_paths = []
    index = 0
    while index < len(specs):
        path = os.path.join(directory, "frame_%04d.png" % index)
        draw_frame(specs[index], path)
        frame_paths.append(path)
        index = index + 1

    assemble_gif(frame_paths, sys.argv[1], width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT,
                 frame_durations=durations)
    print("Saved " + sys.argv[1])


if __name__ == "__main__":
    main()
