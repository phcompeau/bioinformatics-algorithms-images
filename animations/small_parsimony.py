"""Animated small parsimony, in the 02-180 lecture style.

Uses the four species from Evolutionary_Trees.pptx slide 179 (chimp, human, seal,
whale) on the same balanced tree. Small parsimony solves each position of the
alignment independently, so the internal strings fill in column by column while
every edge's mismatch count and the running total build up.

The dynamic program's answer is checked at every position against a brute-force
search over all assignments to the internal nodes, and the final score is checked
to be no worse than the assignment printed on the slide.

Run:  python3 small_parsimony.py OUTPUT.gif
"""

import os
import sys
import tempfile

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, DIM, DISC, FAINT, INK, MONO, MONO_BOLD,
                           OPTIMA, OPTIMA_ITALIC, RED, new_axes)
from make_gif import assemble_gif

LEAF_NAMES = ["Chimp", "Human", "Seal", "Whale"]
LEAF_STRINGS = ["ACGTAGGCCT", "ATGTAAGACT", "TCGAGAGCAC", "TCGAAAGCAT"]
ALPHABET = "ACGT"
# The tree of slide 179: root over two internal nodes, each over two leaves.
CHILDREN = {"root": ["left", "right"], "left": ["Chimp", "Human"],
            "right": ["Seal", "Whale"]}
INTERNAL = ["root", "left", "right"]
SLIDE_SCORE = 8

FIGURE_WIDTH = 10.0
FIGURE_HEIGHT = 6.4
RENDER_DPI = 140
OUTPUT_WIDTH = 1400
OUTPUT_HEIGHT = 896

POSITIONS = {
    "root": (5.0, 5.15),
    "left": (2.675, 3.45), "right": (7.325, 3.45),
    "Chimp": (1.35, 1.75), "Human": (4.0, 1.75),
    "Seal": (6.0, 1.75), "Whale": (8.65, 1.75),
}
CHAR_ADVANCE = 0.145
STRING_SIZE = 14.0
NAME_SIZE = 12.0
EDGE_SIZE = 14.0
NODE_RADIUS = 0.09
SCORE_Y = 0.52
SCORE_SIZE = 26.0

COLUMN_MS = 1450
FIRST_HOLD_MS = 2800
FINAL_HOLD_MS = 5200


def tree_edges() -> "list[tuple[str, str]]":
    result = []
    for parent in CHILDREN:
        for child in CHILDREN[parent]:
            result.append((parent, child))
    return result


EDGES = tree_edges()
LEAF_OF = {}
index = 0
while index < len(LEAF_NAMES):
    LEAF_OF[LEAF_NAMES[index]] = LEAF_STRINGS[index]
    index = index + 1
LENGTH = len(LEAF_STRINGS[0])


def solve_position(column: int) -> "tuple[dict, int]":
    """Sankoff: minimum mismatches for one alignment column, plus an assignment."""
    scores = {}
    choice = {}
    order = ["left", "right", "root"]
    for node in order:
        scores[node] = {}
        choice[node] = {}
        for symbol in ALPHABET:
            total = 0
            picks = {}
            for child in CHILDREN[node]:
                if child in LEAF_OF:
                    letter = LEAF_OF[child][column]
                    if letter == symbol:
                        best = 0
                    else:
                        best = 1
                    picks[child] = letter
                else:
                    best = None
                    for candidate in ALPHABET:
                        cost = scores[child][candidate]
                        if candidate != symbol:
                            cost = cost + 1
                        if best is None or cost < best:
                            best = cost
                            picks[child] = candidate
                total = total + best
            scores[node][symbol] = total
            choice[node][symbol] = dict(picks)
    best_symbol = ALPHABET[0]
    for symbol in ALPHABET:
        if scores["root"][symbol] < scores["root"][best_symbol]:
            best_symbol = symbol
    assignment = {"root": best_symbol}
    pending = ["root"]
    while len(pending) > 0:
        node = pending.pop()
        for child in CHILDREN[node]:
            if child in LEAF_OF:
                continue
            assignment[child] = choice[node][assignment[node]][child]
            pending.append(child)
    return (assignment, scores["root"][best_symbol])


def brute_force_position(column: int) -> int:
    """Minimum mismatches for one column, by trying every internal assignment."""
    best = None
    for first in ALPHABET:
        for second in ALPHABET:
            for third in ALPHABET:
                labels = {"root": first, "left": second, "right": third}
                total = 0
                for parent, child in EDGES:
                    if child in LEAF_OF:
                        child_symbol = LEAF_OF[child][column]
                    else:
                        child_symbol = labels[child]
                    if labels[parent] != child_symbol:
                        total = total + 1
                if best is None or total < best:
                    best = total
    return best


def solve_all() -> "tuple[dict, list, int]":
    strings = {}
    for node in INTERNAL:
        strings[node] = ""
    per_column = []
    total = 0
    column = 0
    while column < LENGTH:
        assignment, score = solve_position(column)
        for node in INTERNAL:
            strings[node] = strings[node] + assignment[node]
        per_column.append(score)
        total = total + score
        column = column + 1
    return (strings, per_column, total)


INTERNAL_STRINGS, COLUMN_SCORES, TOTAL_SCORE = solve_all()


def edge_mismatches(upto: int) -> dict:
    """Hamming distance along each edge over the first `upto` columns."""
    result = {}
    for parent, child in EDGES:
        if child in LEAF_OF:
            child_string = LEAF_OF[child]
        else:
            child_string = INTERNAL_STRINGS[child]
        parent_string = INTERNAL_STRINGS[parent]
        count = 0
        column = 0
        while column < upto:
            if parent_string[column] != child_string[column]:
                count = count + 1
            column = column + 1
        result[(parent, child)] = count
    return result


def verify() -> "list[str]":
    lines = []

    for string in LEAF_STRINGS:
        assert len(string) == LENGTH, "leaf strings must be the same length"
        for letter in string:
            assert letter in ALPHABET, "unexpected symbol %s" % letter
    lines.append("%d leaf strings of length %d over %s"
                 % (len(LEAF_STRINGS), LENGTH, ALPHABET))

    column = 0
    while column < LENGTH:
        assignment, score = solve_position(column)
        brute = brute_force_position(column)
        assert score == brute, (
            "column %d: the dynamic program says %d, brute force says %d"
            % (column, score, brute))
        check = 0
        for parent, child in EDGES:
            if child in LEAF_OF:
                child_symbol = LEAF_OF[child][column]
            else:
                child_symbol = assignment[child]
            if assignment[parent] != child_symbol:
                check = check + 1
        assert check == score, (
            "column %d: the returned assignment scores %d, not %d"
            % (column, check, score))
        column = column + 1
    lines.append("every column's score matches a brute-force search over all %d"
                 " internal assignments" % (len(ALPHABET) ** len(INTERNAL)))

    assert sum(COLUMN_SCORES) == TOTAL_SCORE, "column scores must sum to the total"
    final = edge_mismatches(LENGTH)
    edge_total = 0
    for edge in final:
        edge_total = edge_total + final[edge]
    assert edge_total == TOTAL_SCORE, (
        "edge mismatches total %d but the score is %d" % (edge_total, TOTAL_SCORE))
    lines.append("the %d edge mismatch counts sum to the parsimony score %d"
                 % (len(EDGES), TOTAL_SCORE))

    assert TOTAL_SCORE <= SLIDE_SCORE, (
        "found %d, worse than the slide's assignment at %d"
        % (TOTAL_SCORE, SLIDE_SCORE))
    lines.append("score %d is no worse than the slide's hand-picked assignment (%d)"
                 % (TOTAL_SCORE, SLIDE_SCORE))

    span = LENGTH * CHAR_ADVANCE
    for name in POSITIONS:
        x = POSITIONS[name][0]
        assert x - span / 2 > 0.15, "%s runs off the left" % name
        assert x + span / 2 < FIGURE_WIDTH - 0.15, "%s runs off the right" % name
    assert POSITIONS["Chimp"][1] - 0.55 > SCORE_Y + 0.3, "leaves collide with the score"
    # Two strings drawn on the same row must not run into each other.
    names = sorted(POSITIONS.keys())
    outer = 0
    while outer < len(names):
        inner = outer + 1
        while inner < len(names):
            first = POSITIONS[names[outer]]
            second = POSITIONS[names[inner]]
            if abs(first[1] - second[1]) < 0.2:
                gap = abs(first[0] - second[0]) - span
                assert gap > 0.25, (
                    "%s and %s overlap: only %.2f in between them"
                    % (names[outer], names[inner], gap))
            inner = inner + 1
        outer = outer + 1
    lines.append("all seven node strings fit and no two on a row overlap")
    return lines


def base_frame(duration: int) -> dict:
    return {"upto": 0, "current": -1, "duration_ms": duration}


def build_specs() -> "list[dict]":
    specs = []
    opening = base_frame(FIRST_HOLD_MS)
    specs.append(opening)
    column = 0
    while column < LENGTH:
        frame = base_frame(COLUMN_MS)
        frame["upto"] = column + 1
        frame["current"] = column
        specs.append(frame)
        column = column + 1
    final = base_frame(FINAL_HOLD_MS)
    final["upto"] = LENGTH
    final["current"] = -1
    specs.append(final)
    return specs


def draw_string(axis: "plt.Axes", name: str, text: str, upto: int,
                current: int) -> None:
    x, y = POSITIONS[name]
    advance = CHAR_ADVANCE
    start = x - len(text) * advance / 2
    column = 0
    while column < len(text):
        if name in LEAF_OF:
            colour = INK
        elif column < upto:
            colour = INK
        else:
            colour = FAINT
        if column == current:
            colour = RED
        axis.text(start + (column + 0.5) * advance, y, text[column],
                  fontsize=STRING_SIZE, color=colour, ha="center", va="center",
                  fontproperties=MONO_BOLD, zorder=6)
        column = column + 1


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)
    upto = spec["upto"]
    counts = edge_mismatches(upto)

    for parent, child in EDGES:
        a = POSITIONS[parent]
        b = POSITIONS[child]
        axis.plot([a[0], b[0]], [a[1] - 0.2, b[1] + 0.28], color=INK,
                  linewidth=1.9, solid_capstyle="round", zorder=3)
        mid_x = (a[0] + b[0]) / 2
        mid_y = (a[1] - 0.2 + b[1] + 0.28) / 2
        if upto > 0:
            axis.text(mid_x + 0.22, mid_y, str(counts[(parent, child)]),
                      fontsize=EDGE_SIZE, color=DIM, ha="left", va="center",
                      fontproperties=MONO_BOLD, zorder=6)

    for name in POSITIONS:
        if name in LEAF_OF:
            draw_string(axis, name, LEAF_OF[name], upto, spec["current"])
            axis.text(POSITIONS[name][0], POSITIONS[name][1] - 0.38, name,
                      fontsize=NAME_SIZE, color=INK, ha="center", va="center",
                      fontproperties=OPTIMA, zorder=6)
        else:
            draw_string(axis, name, INTERNAL_STRINGS[name], upto, spec["current"])

    if upto > 0:
        running = 0
        column = 0
        while column < upto:
            running = running + COLUMN_SCORES[column]
            column = column + 1
        axis.text(FIGURE_WIDTH / 2, SCORE_Y, str(running), fontsize=SCORE_SIZE,
                  color=INK, ha="center", va="center", fontproperties=MONO_BOLD,
                  zorder=7)

    figure.savefig(output_path, facecolor=BACKGROUND)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: small_parsimony.py OUTPUT.gif")
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
    directory = tempfile.mkdtemp(prefix="parsimony_")
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
