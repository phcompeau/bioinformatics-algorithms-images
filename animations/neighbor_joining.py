"""Animated neighbor-joining, in the 02-180 lecture style.

Follows Evolutionary_Trees.pptx slides 145-171: build the neighbor-joining matrix
D* from D, join the pair with the smallest D* entry, set their limb lengths from
the total-distance difference, replace them with a new node, and recurse. With
three taxa left every pair is a neighbor, which is the base case.

The additive matrix is recovered from the deck (its TotalDistance column pins it
down), and every intermediate matrix and limb length is asserted against the
numbers printed on the slides. The finished tree is also checked to reproduce D
exactly, which is the whole point of neighbor-joining on an additive matrix.

Run:  python3 neighbor_joining.py OUTPUT.gif
"""

import os
import sys
import tempfile

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, DIM, DISC, INK, MONO, MONO_BOLD, RED,
                           OPTIMA_ITALIC, ease, new_axes)
from make_gif import assemble_gif

TAXA = ["v1", "v2", "v3", "v4"]
DISTANCES = [
    [0, 13, 21, 22],
    [13, 0, 12, 13],
    [21, 12, 0, 13],
    [22, 13, 13, 0],
]
# Printed on slide 145.
PUBLISHED_TOTAL = [56, 38, 46, 48]
PUBLISHED_STAR = [
    [0, -68, -60, -60],
    [-68, 0, -60, -60],
    [-60, -60, 0, -68],
    [-60, -60, -68, 0],
]
# Printed on slide 171.
PUBLISHED_LIMBS = {"v1": 11, "v2": 2, "v3": 6, "v4": 7, "m": 4}

FIGURE_WIDTH = 10.4
FIGURE_HEIGHT = 5.8
RENDER_DPI = 135
OUTPUT_WIDTH = 1404
OUTPUT_HEIGHT = 783

CELL = 0.70
MATRIX_LEFT = 0.60
MATRIX_TOP = 4.05
HEADER_SIZE = 13.0
VALUE_SIZE = 14.0
TITLE_SIZE = 16.0

LEAF_RADIUS = 0.19
LEAF_SIZE = 11.0
NODE_RADIUS = 0.10
LIMB_SIZE = 13.0

CAPTION_Y = 5.36
LABEL_SIZE = 15.0

HOLD_MS = 3000
GROW_FRAMES = 9
GROW_MS = 110
FINAL_HOLD_MS = 5400

# Schematic tree positions, laid out like slide 171.
POSITIONS = {
    "v1": (6.35, 3.60), "v2": (6.35, 1.55),
    "m": (7.45, 2.58), "c": (8.65, 2.58),
    "v3": (9.75, 3.60), "v4": (9.75, 1.55),
}
TREE_EDGES = [("v1", "m"), ("v2", "m"), ("m", "c"), ("v3", "c"), ("v4", "c")]


def total_distance(names: "list[str]", matrix: dict) -> dict:
    result = {}
    for name in names:
        total = 0.0
        for other in names:
            total = total + matrix[(name, other)]
        result[name] = total
    return result


def neighbor_matrix(names: "list[str]", matrix: dict) -> dict:
    """D*(i,j) = (n-2) D(i,j) - TotalDistance(i) - TotalDistance(j)."""
    totals = total_distance(names, matrix)
    count = len(names)
    result = {}
    for first in names:
        for second in names:
            if first == second:
                result[(first, second)] = 0.0
            else:
                result[(first, second)] = ((count - 2) * matrix[(first, second)]
                                           - totals[first] - totals[second])
    return result


def run_neighbor_joining() -> "list[dict]":
    """Record each round: D, D*, the joined pair, limbs, and the reduced matrix."""
    names = list(TAXA)
    matrix = {}
    row = 0
    while row < len(TAXA):
        column = 0
        while column < len(TAXA):
            matrix[(TAXA[row], TAXA[column])] = float(DISTANCES[row][column])
            column = column + 1
        row = row + 1

    rounds = []
    while len(names) > 3:
        totals = total_distance(names, matrix)
        star = neighbor_matrix(names, matrix)
        best = None
        first_index = 0
        while first_index < len(names):
            second_index = first_index + 1
            while second_index < len(names):
                value = star[(names[first_index], names[second_index])]
                if best is None or value < best[0]:
                    best = (value, names[first_index], names[second_index])
                second_index = second_index + 1
            first_index = first_index + 1
        value, left, right = best
        count = len(names)
        delta = (totals[left] - totals[right]) / (count - 2)
        limb_left = (matrix[(left, right)] + delta) / 2
        limb_right = (matrix[(left, right)] - delta) / 2

        reduced = []
        for name in names:
            if name != left and name != right:
                reduced.append(name)
        new_name = "m"
        for name in reduced:
            joined = (matrix[(left, name)] + matrix[(right, name)]
                      - matrix[(left, right)]) / 2
            matrix[(new_name, name)] = joined
            matrix[(name, new_name)] = joined
        matrix[(new_name, new_name)] = 0.0
        reduced.append(new_name)

        rounds.append({"names": list(names), "matrix": dict(matrix),
                       "totals": dict(totals), "star": dict(star),
                       "pair": (left, right), "delta": delta,
                       "limbs": {left: limb_left, right: limb_right},
                       "after_names": list(reduced)})
        names = reduced

    # Base case: three taxa, every pair a neighbor, so hang a star.
    totals = total_distance(names, matrix)
    star = neighbor_matrix(names, matrix)
    limbs = {}
    for name in names:
        others = []
        for other in names:
            if other != name:
                others.append(other)
        limbs[name] = (matrix[(name, others[0])] + matrix[(name, others[1])]
                       - matrix[(others[0], others[1])]) / 2
    rounds.append({"names": list(names), "matrix": dict(matrix),
                   "totals": dict(totals), "star": dict(star), "pair": (),
                   "delta": 0.0, "limbs": dict(limbs), "after_names": []})
    return rounds


ROUNDS = run_neighbor_joining()


def all_limbs() -> dict:
    result = {}
    for entry in ROUNDS:
        for name in entry["limbs"]:
            result[name] = entry["limbs"][name]
    return result


LIMBS = all_limbs()


def edge_lengths() -> dict:
    """Length of each tree edge, keyed both ways for easy lookup."""
    result = {}
    for a, b in TREE_EDGES:
        if a == "m" and b == "c":
            length = LIMBS["m"]
        elif a in LIMBS:
            length = LIMBS[a]
        else:
            length = LIMBS[b]
        result[(a, b)] = length
        result[(b, a)] = length
    return result


EDGE_LENGTH = edge_lengths()


def tree_distance(start: str, goal: str) -> float:
    """Path length between two nodes of the finished tree."""
    neighbours = {}
    for a, b in TREE_EDGES:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    stack = [(start, 0.0, "")]
    while len(stack) > 0:
        node, distance, came = stack.pop()
        if node == goal:
            return distance
        for other in neighbours[node]:
            if other == came:
                continue
            stack.append((other, distance + EDGE_LENGTH[(node, other)], node))
    raise ValueError("no path from %s to %s" % (start, goal))


def verify() -> "list[str]":
    lines = []
    size = len(TAXA)

    first = ROUNDS[0]
    totals = []
    for name in TAXA:
        totals.append(first["totals"][name])
    assert totals == [float(value) for value in PUBLISHED_TOTAL], (
        "TotalDistance is %s but slide 145 prints %s" % (totals, PUBLISHED_TOTAL))
    lines.append("TotalDistance is %s, matching slide 145"
                 % " ".join("%g" % value for value in totals))

    row = 0
    while row < size:
        column = 0
        while column < size:
            got = first["star"][(TAXA[row], TAXA[column])]
            assert got == float(PUBLISHED_STAR[row][column]), (
                "D*(%s,%s) is %g but the slide prints %g"
                % (TAXA[row], TAXA[column], got, PUBLISHED_STAR[row][column]))
            column = column + 1
        row = row + 1
    lines.append("the whole neighbor-joining matrix D* matches slide 145")

    assert first["pair"] == ("v1", "v2"), (
        "expected v1 and v2 to join first, got %s" % (first["pair"],))
    assert first["delta"] == 9, "delta should be 9, got %g" % first["delta"]
    lines.append("v1 and v2 join first at D* = -68, with delta = 9")

    reduced = ROUNDS[1]
    assert sorted(reduced["names"]) == ["m", "v3", "v4"], (
        "the reduced matrix should be on m, v3, v4")
    assert reduced["matrix"][("m", "v3")] == 10, "D'(m,v3) should be 10"
    assert reduced["matrix"][("m", "v4")] == 11, "D'(m,v4) should be 11"
    lines.append("the reduced matrix has D'(m,v3) = 10 and D'(m,v4) = 11")

    values = []
    for name in reduced["names"]:
        for other in reduced["names"]:
            if name != other:
                values.append(reduced["star"][(name, other)])
    assert len(set(values)) == 1, "with three taxa every D* entry should be equal"
    lines.append("with three taxa left every D* entry is %g: all of them are neighbors"
                 % values[0])

    for name in PUBLISHED_LIMBS:
        assert LIMBS[name] == PUBLISHED_LIMBS[name], (
            "limb %s is %g but slide 171 prints %g"
            % (name, LIMBS[name], PUBLISHED_LIMBS[name]))
    lines.append("all five limb lengths match slide 171: %s"
                 % " ".join("%s=%g" % (name, LIMBS[name])
                            for name in ["v1", "v2", "v3", "v4", "m"]))

    row = 0
    while row < size:
        column = 0
        while column < size:
            if row != column:
                through = tree_distance(TAXA[row], TAXA[column])
                assert abs(through - DISTANCES[row][column]) < 1e-9, (
                    "tree gives d(%s,%s) = %g but D says %g"
                    % (TAXA[row], TAXA[column], through, DISTANCES[row][column]))
            column = column + 1
        row = row + 1
    lines.append("the finished tree reproduces every entry of D exactly")

    width = MATRIX_LEFT + (size + 2) * CELL
    assert width < POSITIONS["v1"][0] - 0.3, "matrix collides with the tree"
    assert POSITIONS["v3"][0] + 0.4 < FIGURE_WIDTH, "tree runs off the right"
    assert POSITIONS["v1"][1] + 0.4 < CAPTION_Y, "tree runs into the caption"
    lines.append("matrix ends at %.2f in, tree starts at %.2f in"
                 % (width, POSITIONS["v1"][0]))
    return lines


def base_frame(duration: int) -> dict:
    return {"names": [], "values": {}, "totals": {}, "title": "", "highlight": (),
            "built": [], "growth": 1.0, "caption": "", "duration_ms": duration}


def build_specs() -> "list[dict]":
    specs = []
    first = ROUNDS[0]

    show_d = base_frame(HOLD_MS)
    show_d["names"] = list(first["names"])
    show_d["values"] = dict(first["matrix"])
    show_d["totals"] = dict(first["totals"])
    show_d["title"] = "D"
    show_d["caption"] = "Start from the distance matrix and its row totals"
    specs.append(show_d)

    show_star = base_frame(HOLD_MS)
    show_star["names"] = list(first["names"])
    show_star["values"] = dict(first["star"])
    show_star["totals"] = dict(first["totals"])
    show_star["title"] = "D*"
    show_star["highlight"] = first["pair"]
    show_star["caption"] = "D*(i,j) = (n-2) D(i,j) - TotalDistance(i) - TotalDistance(j)"
    specs.append(show_star)

    pick = base_frame(HOLD_MS)
    pick["names"] = list(first["names"])
    pick["values"] = dict(first["star"])
    pick["totals"] = dict(first["totals"])
    pick["title"] = "D*"
    pick["highlight"] = first["pair"]
    pick["caption"] = ("Its smallest entry is -68, so v1 and v2 are neighbors")
    specs.append(pick)

    step = 1
    while step <= GROW_FRAMES:
        grow = base_frame(GROW_MS)
        grow["names"] = list(first["names"])
        grow["values"] = dict(first["star"])
        grow["totals"] = dict(first["totals"])
        grow["title"] = "D*"
        grow["highlight"] = first["pair"]
        grow["built"] = [("v1", "m"), ("v2", "m")]
        grow["growth"] = ease(step / GROW_FRAMES)
        grow["caption"] = "Limb lengths come from delta = 9: v1 gets 11, v2 gets 2"
        specs.append(grow)
        step = step + 1

    hung = base_frame(HOLD_MS)
    hung["names"] = list(first["names"])
    hung["values"] = dict(first["star"])
    hung["totals"] = dict(first["totals"])
    hung["title"] = "D*"
    hung["built"] = [("v1", "m"), ("v2", "m")]
    hung["caption"] = "Limb lengths come from delta = 9: v1 gets 11, v2 gets 2"
    specs.append(hung)

    reduced = ROUNDS[1]
    show_reduced = base_frame(HOLD_MS)
    show_reduced["names"] = list(reduced["names"])
    show_reduced["values"] = dict(reduced["matrix"])
    show_reduced["totals"] = dict(reduced["totals"])
    show_reduced["title"] = "D'"
    show_reduced["built"] = [("v1", "m"), ("v2", "m")]
    show_reduced["caption"] = "Replace v1 and v2 by a node m and recurse"
    specs.append(show_reduced)

    show_reduced_star = base_frame(HOLD_MS)
    show_reduced_star["names"] = list(reduced["names"])
    show_reduced_star["values"] = dict(reduced["star"])
    show_reduced_star["totals"] = dict(reduced["totals"])
    show_reduced_star["title"] = "D'*"
    show_reduced_star["built"] = [("v1", "m"), ("v2", "m")]
    show_reduced_star["caption"] = "With three taxa left, every entry ties: they are all neighbors"
    specs.append(show_reduced_star)

    step = 1
    while step <= GROW_FRAMES:
        grow = base_frame(GROW_MS)
        grow["names"] = list(reduced["names"])
        grow["values"] = dict(reduced["star"])
        grow["totals"] = dict(reduced["totals"])
        grow["title"] = "D'*"
        grow["built"] = list(TREE_EDGES)
        grow["growth"] = ease(step / GROW_FRAMES)
        grow["caption"] = "Hang the last three limbs from a single node"
        specs.append(grow)
        step = step + 1

    # Bring D back for the last beat so the tree can be checked against it.
    final = base_frame(FINAL_HOLD_MS)
    final["names"] = list(first["names"])
    final["values"] = dict(first["matrix"])
    final["totals"] = dict(first["totals"])
    final["title"] = "D"
    final["built"] = list(TREE_EDGES)
    final["caption"] = "The tree is additive: it reproduces every distance in D"
    specs.append(final)
    return specs


def draw_matrix(axis: "plt.Axes", spec: dict) -> None:
    names = spec["names"]
    if len(names) == 0:
        return
    if spec["title"] != "":
        axis.text(MATRIX_LEFT - 0.1, MATRIX_TOP - (len(names) + 1) * CELL / 2,
                  spec["title"], fontsize=TITLE_SIZE, color=INK, ha="right",
                  va="center", fontproperties=OPTIMA_ITALIC, zorder=5)
    column = 0
    while column < len(names):
        highlighted = names[column] in spec["highlight"]
        if highlighted:
            colour = RED
        else:
            colour = INK
        axis.text(MATRIX_LEFT + (column + 1.5) * CELL, MATRIX_TOP, names[column],
                  fontsize=HEADER_SIZE, color=colour, ha="center", va="center",
                  fontproperties=OPTIMA_ITALIC, zorder=5)
        axis.text(MATRIX_LEFT + 0.5 * CELL, MATRIX_TOP - (column + 1) * CELL,
                  names[column], fontsize=HEADER_SIZE, color=colour, ha="center",
                  va="center", fontproperties=OPTIMA_ITALIC, zorder=5)
        column = column + 1
    axis.text(MATRIX_LEFT + (len(names) + 1.6) * CELL, MATRIX_TOP, "total",
              fontsize=HEADER_SIZE - 1, color=DIM, ha="center", va="center",
              fontproperties=OPTIMA_ITALIC, zorder=5)

    row = 0
    while row < len(names):
        column = 0
        while column < len(names):
            value = spec["values"][(names[row], names[column])]
            pair_hit = (names[row] in spec["highlight"]
                        and names[column] in spec["highlight"]
                        and names[row] != names[column])
            if pair_hit:
                colour = RED
                font = MONO_BOLD
            elif row == column:
                colour = DIM
                font = MONO
            else:
                colour = INK
                font = MONO
            axis.text(MATRIX_LEFT + (column + 1.5) * CELL,
                      MATRIX_TOP - (row + 1) * CELL, "%g" % value,
                      fontsize=VALUE_SIZE, color=colour, ha="center", va="center",
                      fontproperties=font, zorder=5)
            column = column + 1
        axis.text(MATRIX_LEFT + (len(names) + 1.6) * CELL,
                  MATRIX_TOP - (row + 1) * CELL,
                  "%g" % spec["totals"][names[row]], fontsize=VALUE_SIZE,
                  color=DIM, ha="center", va="center", fontproperties=MONO,
                  zorder=5)
        row = row + 1


def draw_tree(axis: "plt.Axes", spec: dict) -> None:
    built = spec["built"]
    index = 0
    while index < len(built):
        edge = built[index]
        if index >= len(built) - 3 and spec["growth"] < 1.0:
            reach = spec["growth"]
        else:
            reach = 1.0
        if edge in [("v1", "m"), ("v2", "m")] and spec["growth"] < 1.0 and len(built) == 2:
            reach = spec["growth"]
        start = POSITIONS[edge[1]]
        end = POSITIONS[edge[0]]
        x = start[0] + (end[0] - start[0]) * reach
        y = start[1] + (end[1] - start[1]) * reach
        axis.plot([start[0], x], [start[1], y], color=INK, linewidth=1.9,
                  solid_capstyle="round", zorder=4)
        if reach > 0.98:
            label = LIMBS.get(edge[0], LIMBS.get(edge[1], 0))
            if edge == ("m", "c"):
                label = LIMBS["m"]
                axis.text((start[0] + end[0]) / 2, start[1] + 0.16, "%g" % label,
                          fontsize=LIMB_SIZE, color=INK, ha="center", va="bottom",
                          fontproperties=MONO, zorder=6)
            else:
                axis.text((start[0] + end[0]) / 2 + 0.14,
                          (start[1] + end[1]) / 2, "%g" % label,
                          fontsize=LIMB_SIZE, color=INK, ha="left", va="center",
                          fontproperties=MONO, zorder=6)
        index = index + 1

    drawn = set()
    for edge in built:
        drawn.add(edge[0])
        drawn.add(edge[1])
    for name in drawn:
        position = POSITIONS[name]
        if name in TAXA:
            disc = patches.Circle(position, LEAF_RADIUS, facecolor=DISC,
                                  edgecolor="none", zorder=5)
            axis.add_patch(disc)
            axis.text(position[0], position[1], name, fontsize=LEAF_SIZE,
                      color="white", ha="center", va="center",
                      fontproperties=OPTIMA_ITALIC, zorder=6)
        else:
            dot = patches.Circle(position, NODE_RADIUS, facecolor=INK,
                                 edgecolor="none", zorder=5)
            axis.add_patch(dot)
            if name == "m":
                axis.text(position[0] - 0.06, position[1] - 0.26, "m",
                          fontsize=LIMB_SIZE, color=INK, ha="center", va="top",
                          fontproperties=OPTIMA_ITALIC, zorder=6)


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)
    draw_matrix(axis, spec)
    draw_tree(axis, spec)
    if spec["caption"] != "":
        axis.text(FIGURE_WIDTH / 2, CAPTION_Y, spec["caption"], fontsize=LABEL_SIZE,
                  color=INK, ha="center", va="center", fontproperties=OPTIMA_ITALIC,
                  zorder=7)
    figure.savefig(output_path, facecolor=BACKGROUND)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: neighbor_joining.py OUTPUT.gif")
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
    directory = tempfile.mkdtemp(prefix="nj_")
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
