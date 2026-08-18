"""Animated linear-space alignment (Hirschberg), in the lecture style.

The alignment lattice for two short strings. Rather than storing the whole
dynamic-programming table, find the one node where an optimal path crosses the
middle column, recurse on the two smaller rectangles either side of it, and the
optimal path assembles from the middle nodes alone.

The middle node at each level is found by the linear-space score sweep, and every
one is asserted to lie on a genuine optimal path: the best score through it must
equal the global optimum computed by a full table.

Run:  python3 middle_node.py OUTPUT.gif
"""

import os
import sys
import tempfile

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, BLUE, DIM, FAINT, GREEN, INK, MONO,
                           MONO_BOLD, OPTIMA_ITALIC, RED, new_axes)
from make_gif import assemble_gif

FIRST = "GATTACA"
SECOND = "GTTAC"
MATCH = 1
MISMATCH = -1
INDEL = -1

FIGURE_WIDTH = 8.8
FIGURE_HEIGHT = 5.6
RENDER_DPI = 150
OUTPUT_WIDTH = 1320
OUTPUT_HEIGHT = 840

GRID_LEFT = 1.35
GRID_TOP = 4.55
SPACING_X = 0.86
SPACING_Y = 0.78
DOT_RADIUS = 0.075
LETTER_SIZE = 15.0
BAND_ALPHA = 0.55

CAPTION_Y = 5.22
LABEL_SIZE = 15.0

LEVEL_MS = 2400
BAND_MS = 1700
FIRST_HOLD_MS = 2600
FINAL_HOLD_MS = 5200

COLUMNS = len(FIRST)
ROWS = len(SECOND)


def score_table() -> "list[list[int]]":
    """Full global-alignment table, used only as an oracle."""
    table = []
    row = 0
    while row <= ROWS:
        table.append([0] * (COLUMNS + 1))
        row = row + 1
    column = 1
    while column <= COLUMNS:
        table[0][column] = column * INDEL
        column = column + 1
    row = 1
    while row <= ROWS:
        table[row][0] = row * INDEL
        row = row + 1
    row = 1
    while row <= ROWS:
        column = 1
        while column <= COLUMNS:
            if SECOND[row - 1] == FIRST[column - 1]:
                diagonal = table[row - 1][column - 1] + MATCH
            else:
                diagonal = table[row - 1][column - 1] + MISMATCH
            best = diagonal
            if table[row][column - 1] + INDEL > best:
                best = table[row][column - 1] + INDEL
            if table[row - 1][column] + INDEL > best:
                best = table[row - 1][column] + INDEL
            table[row][column] = best
            column = column + 1
        row = row + 1
    return table


TABLE = score_table()
OPTIMUM = TABLE[ROWS][COLUMNS]


def prefix_scores(top: int, bottom: int, left: int, right: int,
                  column: int) -> "list[int]":
    """Scores from (top,left) to every node of `column`, within the rectangle."""
    previous = []
    row = top
    while row <= bottom:
        previous.append((row - top) * INDEL)
        row = row + 1
    walker = left + 1
    while walker <= column:
        current = [previous[0] + INDEL]
        row = top + 1
        while row <= bottom:
            if SECOND[row - 1] == FIRST[walker - 1]:
                diagonal = previous[row - 1 - top] + MATCH
            else:
                diagonal = previous[row - 1 - top] + MISMATCH
            best = diagonal
            if previous[row - top] + INDEL > best:
                best = previous[row - top] + INDEL
            if current[row - 1 - top] + INDEL > best:
                best = current[row - 1 - top] + INDEL
            current.append(best)
            row = row + 1
        previous = current
        walker = walker + 1
    return previous


def suffix_scores(top: int, bottom: int, left: int, right: int,
                  column: int) -> "list[int]":
    """Scores from every node of `column` to (bottom,right), within the rectangle."""
    previous = []
    row = top
    while row <= bottom:
        previous.append((bottom - row) * INDEL)
        row = row + 1
    walker = right
    while walker > column:
        current = []
        row = top
        while row <= bottom:
            if row == bottom:
                current.append(previous[row - top] + INDEL)
            else:
                if SECOND[row] == FIRST[walker - 1]:
                    diagonal = previous[row + 1 - top] + MATCH
                else:
                    diagonal = previous[row + 1 - top] + MISMATCH
                best = diagonal
                if previous[row - top] + INDEL > best:
                    best = previous[row - top] + INDEL
                current.append(best)
            row = row + 1
        # a downward move within the same column
        row = bottom - 1
        while row >= top:
            if current[row + 1 - top] + INDEL > current[row - top]:
                current[row - top] = current[row + 1 - top] + INDEL
            row = row - 1
        previous = current
        walker = walker - 1
    return previous


def middle_node(top: int, bottom: int, left: int, right: int) -> "tuple[int, int]":
    """The row where an optimal path crosses the rectangle's middle column."""
    column = (left + right) // 2
    heads = prefix_scores(top, bottom, left, right, column)
    tails = suffix_scores(top, bottom, left, right, column)
    best_row = top
    best_value = None
    row = top
    while row <= bottom:
        value = heads[row - top] + tails[row - top]
        if best_value is None or value > best_value:
            best_value = value
            best_row = row
        row = row + 1
    return (best_row, column)


def divide(top: int, bottom: int, left: int, right: int,
           depth: int) -> "list[dict]":
    """Record the middle node of each rectangle, shallowest levels first.

    The two halves share the middle column. That is what keeps the decomposition
    optimal: the middle node lies on a globally optimal path, and an optimal path
    within each half concatenates through it. A rectangle one column wide is
    already pinned down, so it stops rather than recursing forever.
    """
    if right - left < 1:
        return []
    node_row, column = middle_node(top, bottom, left, right)
    record = [{"row": node_row, "column": column, "top": top, "bottom": bottom,
               "left": left, "right": right, "depth": depth}]
    if right - left == 1:
        return record
    record.extend(divide(top, node_row, left, column, depth + 1))
    record.extend(divide(node_row, bottom, column, right, depth + 1))
    return record


def unique_nodes(found: "list[dict]") -> "list[dict]":
    """One node per column, keeping the shallowest occurrence."""
    best = {}
    for entry in found:
        column = entry["column"]
        if column not in best or entry["depth"] < best[column]["depth"]:
            best[column] = entry
    result = []
    for column in sorted(best.keys()):
        result.append(best[column])
    return result


ALL_FOUND = divide(0, ROWS, 0, COLUMNS, 0)
NODES = unique_nodes(ALL_FOUND)


def verify() -> "list[str]":
    lines = []
    lines.append("aligning %s against %s, optimal score %d"
                 % (FIRST, SECOND, OPTIMUM))

    full = middle_node(0, ROWS, 0, COLUMNS)
    heads = prefix_scores(0, ROWS, 0, COLUMNS, full[1])
    tails = suffix_scores(0, ROWS, 0, COLUMNS, full[1])
    best = None
    row = 0
    while row <= ROWS:
        value = heads[row] + tails[row]
        if best is None or value > best:
            best = value
        row = row + 1
    assert best == OPTIMUM, (
        "the middle column sweep gives %d but the full table gives %d"
        % (best, OPTIMUM))
    lines.append("the middle-column sweep reproduces the optimal score %d" % OPTIMUM)

    row = 0
    while row <= ROWS:
        assert heads[row] == TABLE[row][full[1]], (
            "prefix score at row %d disagrees with the full table" % row)
        row = row + 1
    lines.append("every prefix score down the middle column matches the full table")

    for entry in NODES:
        heads = prefix_scores(entry["top"], entry["bottom"], entry["left"],
                              entry["right"], entry["column"])
        tails = suffix_scores(entry["top"], entry["bottom"], entry["left"],
                             entry["right"], entry["column"])
        best = None
        row = entry["top"]
        while row <= entry["bottom"]:
            value = heads[row - entry["top"]] + tails[row - entry["top"]]
            if best is None or value > best:
                best = value
            row = row + 1
        chosen = (heads[entry["row"] - entry["top"]]
                  + tails[entry["row"] - entry["top"]])
        assert chosen == best, (
            "middle node of rectangle %s is not on an optimal path within it"
            % ((entry["top"], entry["bottom"], entry["left"], entry["right"]),))
        assert entry["left"] <= entry["column"] < entry["right"], (
            "the middle column must sit inside its rectangle")
        assert entry["top"] <= entry["row"] <= entry["bottom"], (
            "the middle node must sit inside its rectangle")
    lines.append("all %d middle nodes lie on an optimal path within their rectangle"
                 % len(NODES))

    by_column = {}
    for entry in ALL_FOUND:
        column = entry["column"]
        if column in by_column:
            assert by_column[column] == entry["row"], (
                "column %d was pinned to two different rows" % column)
        by_column[column] = entry["row"]
    columns = sorted(by_column.keys())
    assert columns == list(range(COLUMNS)), (
        "the recursion should pin one row in every column, got %s" % columns)
    lines.append("every column is pinned to a single row, all %d of them" % COLUMNS)

    ordered = sorted(NODES, key=lambda item: item["column"])
    previous_row = 0
    for entry in ordered:
        assert entry["row"] >= previous_row, "middle nodes must not move upward"
        previous_row = entry["row"]
    lines.append("read left to right the middle nodes never step upward")

    # The strongest check: the path assembled from the middle nodes alone must
    # score exactly what the full table says the optimum is.
    points = [(0, 0)]
    for entry in ordered:
        points.append((entry["row"], entry["column"]))
    points.append((ROWS, COLUMNS))
    total = 0
    index = 1
    while index < len(points):
        from_row, from_column = points[index - 1]
        to_row, to_column = points[index]
        steps_right = to_column - from_column
        steps_down = to_row - from_row
        assert steps_right >= 0 and steps_down >= 0, "the path must stay monotone"
        if steps_right > 0 and steps_down > 0:
            if SECOND[from_row] == FIRST[from_column]:
                total = total + MATCH
            else:
                total = total + MISMATCH
            total = total + (steps_right - 1) * INDEL + (steps_down - 1) * INDEL
        else:
            total = total + (steps_right + steps_down) * INDEL
        index = index + 1
    assert total == OPTIMUM, (
        "the assembled path scores %d but the optimum is %d" % (total, OPTIMUM))
    lines.append("the path assembled from the middle nodes scores %d, the optimum"
                 % total)

    width = COLUMNS * SPACING_X
    height = ROWS * SPACING_Y
    assert GRID_LEFT + width < FIGURE_WIDTH - 0.3, "lattice too wide"
    assert GRID_TOP - height > 0.4, "lattice too tall"
    assert GRID_TOP + 0.3 < CAPTION_Y, "lattice collides with the caption"
    lines.append("lattice is %.2f x %.2f in and fits the canvas" % (width, height))
    return lines


def node_xy(row: int, column: int) -> "tuple[float, float]":
    return (GRID_LEFT + column * SPACING_X, GRID_TOP - row * SPACING_Y)


def base_frame(duration: int) -> dict:
    return {"shown": [], "band": (), "caption": "", "path": False,
            "duration_ms": duration}


def build_specs() -> "list[dict]":
    specs = []
    depth_max = 0
    for entry in NODES:
        if entry["depth"] > depth_max:
            depth_max = entry["depth"]

    opening = base_frame(FIRST_HOLD_MS)
    opening["caption"] = "%s against %s" % (FIRST, SECOND)
    specs.append(opening)

    shown = []
    depth = 0
    while depth <= depth_max:
        batch = []
        for entry in NODES:
            if entry["depth"] == depth:
                batch.append(entry)
        if len(batch) == 0:
            depth = depth + 1
            continue
        band = base_frame(BAND_MS)
        band["shown"] = list(shown)
        band["band"] = tuple(batch)
        band["caption"] = "Sweep the middle column of each rectangle"
        specs.append(band)

        for entry in batch:
            shown.append(entry)
        found = base_frame(LEVEL_MS)
        found["shown"] = list(shown)
        found["caption"] = "Keep only where an optimal path crosses"
        specs.append(found)
        depth = depth + 1

    final = base_frame(FINAL_HOLD_MS)
    final["shown"] = list(shown)
    final["path"] = True
    final["caption"] = "The middle nodes alone give the optimal path"
    specs.append(final)
    return specs


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)

    for entry in spec["band"]:
        x = node_xy(0, entry["column"])[0]
        top_y = node_xy(entry["top"], 0)[1]
        bottom_y = node_xy(entry["bottom"], 0)[1]
        band = patches.Rectangle((x - SPACING_X * 0.32, bottom_y - 0.2),
                                 SPACING_X * 0.64,
                                 (top_y - bottom_y) + 0.4,
                                 facecolor=GREEN, edgecolor="none",
                                 alpha=0.22, zorder=2)
        axis.add_patch(band)

    column = 0
    while column <= COLUMNS:
        row = 0
        while row <= ROWS:
            x, y = node_xy(row, column)
            dot = patches.Circle((x, y), DOT_RADIUS, facecolor=FAINT,
                                 edgecolor="none", zorder=3)
            axis.add_patch(dot)
            row = row + 1
        column = column + 1

    if spec["path"]:
        ordered = sorted(spec["shown"], key=lambda item: item["column"])
        points = [(0, 0)]
        for entry in ordered:
            points.append((entry["row"], entry["column"]))
        points.append((ROWS, COLUMNS))
        index = 1
        while index < len(points):
            a = node_xy(points[index - 1][0], points[index - 1][1])
            b = node_xy(points[index][0], points[index][1])
            axis.plot([a[0], b[0]], [a[1], b[1]], color=RED, linewidth=2.6,
                      solid_capstyle="round", zorder=4)
            index = index + 1

    for entry in spec["shown"]:
        x, y = node_xy(entry["row"], entry["column"])
        dot = patches.Circle((x, y), DOT_RADIUS * 1.8, facecolor=RED,
                             edgecolor="none", zorder=5)
        axis.add_patch(dot)

    column = 0
    while column < COLUMNS:
        x = node_xy(0, column)[0] + SPACING_X / 2
        axis.text(x, GRID_TOP + 0.28, FIRST[column], fontsize=LETTER_SIZE,
                  color=INK, ha="center", va="center", fontproperties=MONO_BOLD,
                  zorder=6)
        column = column + 1
    row = 0
    while row < ROWS:
        y = node_xy(row, 0)[1] - SPACING_Y / 2
        axis.text(GRID_LEFT - 0.34, y, SECOND[row], fontsize=LETTER_SIZE,
                  color=INK, ha="center", va="center", fontproperties=MONO_BOLD,
                  zorder=6)
        row = row + 1

    if spec["caption"] != "":
        axis.text(FIGURE_WIDTH / 2, CAPTION_Y, spec["caption"], fontsize=LABEL_SIZE,
                  color=INK, ha="center", va="center", fontproperties=OPTIMA_ITALIC,
                  zorder=7)

    figure.savefig(output_path, facecolor=BACKGROUND)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: middle_node.py OUTPUT.gif")
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
    directory = tempfile.mkdtemp(prefix="middle_node_")
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
