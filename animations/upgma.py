"""Animated UPGMA clustering, in the 02-180 lecture style.

Follows Evolutionary_Trees.pptx slides 110-120: start with one cluster per
species, repeatedly merge the two closest clusters, hang a new node at half
their distance, and replace their rows and columns with a weighted average.

The distance matrix is read from slide 110. The resulting node ages and limb
lengths are asserted against the numbers printed on slide 118, and the tree is
checked for the ultrametric property UPGMA guarantees.

Run:  python3 upgma.py OUTPUT.gif
"""

import os
import sys
import tempfile

import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, DIM, DISC, INK, MONO, MONO_BOLD, RED,
                           OPTIMA, OPTIMA_ITALIC, ease, new_axes)
from make_gif import assemble_gif

LEAVES = ["i", "j", "k", "l"]
# Slide 110's distance matrix.
DISTANCES = [
    [0, 3, 4, 3],
    [3, 0, 4, 5],
    [4, 4, 0, 2],
    [3, 5, 2, 0],
]
# Every number printed on slide 118: three internal node ages plus six limbs.
PUBLISHED_NUMBERS = [0.5, 1, 1, 1, 1, 1.5, 1.5, 1.5, 2]

FIGURE_WIDTH = 10.0
FIGURE_HEIGHT = 5.6
RENDER_DPI = 140
OUTPUT_WIDTH = 1400
OUTPUT_HEIGHT = 784

CELL = 0.62
MATRIX_LEFT = 0.75
MATRIX_TOP = 3.95
HEADER_SIZE = 13.0
VALUE_SIZE = 14.0

TREE_LEFT = 5.85
TREE_LEAF_GAP = 1.05
TREE_BASE = 0.95
TREE_SCALE = 1.30
LEAF_RADIUS = 0.17
LEAF_SIZE = 12.0
LIMB_SIZE = 12.0

CAPTION_Y = 5.18
LABEL_SIZE = 15.0

STEP_HOLD_MS = 2600
GROW_FRAMES = 9
GROW_MS = 110
MERGE_HOLD_MS = 2200
FIRST_HOLD_MS = 3000
FINAL_HOLD_MS = 5200


def run_upgma() -> "list[dict]":
    """Each iteration: the matrix before, the pair merged, and the new node."""
    names = list(LEAVES)
    sizes = {}
    ages = {}
    for name in LEAVES:
        sizes[name] = 1
        ages[name] = 0.0
    matrix = {}
    row = 0
    while row < len(LEAVES):
        column = 0
        while column < len(LEAVES):
            matrix[(LEAVES[row], LEAVES[column])] = float(DISTANCES[row][column])
            column = column + 1
        row = row + 1

    record = []
    while len(names) > 1:
        best = None
        first = 0
        while first < len(names):
            second = first + 1
            while second < len(names):
                value = matrix[(names[first], names[second])]
                if best is None or value < best[0]:
                    best = (value, names[first], names[second])
                second = second + 1
            first = first + 1
        distance, left, right = best
        merged = left + right
        age = distance / 2.0
        step = {"names": list(names),
                "matrix": dict(matrix),
                "pair": (left, right),
                "distance": distance,
                "merged": merged,
                "age": age,
                "children": [(left, age - ages[left]), (right, age - ages[right])]}
        new_names = []
        for name in names:
            if name != left and name != right:
                new_names.append(name)
        new_names.append(merged)
        sizes[merged] = sizes[left] + sizes[right]
        ages[merged] = age
        for name in new_names:
            if name == merged:
                matrix[(merged, merged)] = 0.0
                continue
            weighted = (sizes[left] * matrix[(left, name)]
                        + sizes[right] * matrix[(right, name)])
            value = weighted / (sizes[left] + sizes[right])
            matrix[(merged, name)] = value
            matrix[(name, merged)] = value
        names = new_names
        step["after_names"] = list(names)
        step["after_matrix"] = dict(matrix)
        record.append(step)
    return record


STEPS = run_upgma()


def node_ages() -> dict:
    result = {}
    for name in LEAVES:
        result[name] = 0.0
    for step in STEPS:
        result[step["merged"]] = step["age"]
    return result


AGES = node_ages()


def leaf_slot() -> dict:
    """Left-to-right leaf order, chosen so merged clusters stay adjacent."""
    result = {}
    index = 0
    while index < len(LEAVES):
        result[LEAVES[index]] = float(index)
        index = index + 1
    return result


SLOT = leaf_slot()


def cluster_slot(name: str) -> float:
    total = 0.0
    count = 0
    for character in name:
        total = total + SLOT[character]
        count = count + 1
    return total / count


def verify() -> "list[str]":
    lines = []
    size = len(LEAVES)

    row = 0
    while row < size:
        assert DISTANCES[row][row] == 0, "diagonal must be zero"
        column = 0
        while column < size:
            assert DISTANCES[row][column] == DISTANCES[column][row], (
                "matrix must be symmetric")
            column = column + 1
        row = row + 1
    lines.append("distance matrix is symmetric with a zero diagonal")

    assert len(STEPS) == size - 1, "n-1 merges for n leaves"
    lines.append("UPGMA performed %d merges for %d species" % (len(STEPS), size))

    numbers = []
    for step in STEPS:
        numbers.append(step["age"])
        for child, limb in step["children"]:
            numbers.append(limb)
    numbers.sort()
    assert numbers == sorted(PUBLISHED_NUMBERS), (
        "computed %s but slide 118 prints %s" % (numbers, sorted(PUBLISHED_NUMBERS)))
    lines.append("node ages and limb lengths are %s, matching slide 118"
                 % " ".join(("%g" % value) for value in numbers))

    merge_order = []
    for step in STEPS:
        merge_order.append("{%s}" % ",".join(step["pair"]))
    lines.append("merge order is %s" % " then ".join(merge_order))

    for step in STEPS:
        for child, limb in step["children"]:
            assert limb > 0, "limb lengths must be positive"
            assert abs(AGES[child] + limb - step["age"]) < 1e-9, (
                "limb does not reach its parent's age")
    lines.append("every limb reaches exactly from its child's age to its parent's")

    # UPGMA always yields an ultrametric tree: all leaves are equidistant from
    # the root through the tree.
    root = STEPS[len(STEPS) - 1]["merged"]
    for leaf in LEAVES:
        depth = 0.0
        name = leaf
        while name != root:
            parent = ""
            for step in STEPS:
                if name in step["pair"]:
                    parent = step["merged"]
                    for child, limb in step["children"]:
                        if child == name:
                            depth = depth + limb
            assert parent != "", "leaf %s has no path to the root" % leaf
            name = parent
        assert abs(depth - AGES[root]) < 1e-9, (
            "leaf %s sits %g from the root, not %g" % (leaf, depth, AGES[root]))
    lines.append("all %d leaves are %g from the root: the tree is ultrametric"
                 % (size, AGES[root]))

    top = TREE_BASE + AGES[root] * TREE_SCALE
    assert top + 0.3 < CAPTION_Y, "tree runs into the caption"
    width = MATRIX_LEFT + (size + 1) * CELL
    assert width < TREE_LEFT - 0.3, "matrix collides with the tree"
    right = TREE_LEFT + (size - 1) * TREE_LEAF_GAP
    assert right + 0.4 < FIGURE_WIDTH, "tree runs off the right"
    lines.append("matrix ends at %.2f in, tree spans %.2f to %.2f in"
                 % (width, TREE_LEFT, right))
    return lines


def leaf_x(name: str) -> float:
    return TREE_LEFT + cluster_slot(name) * TREE_LEAF_GAP


def height_y(age: float) -> float:
    return TREE_BASE + age * TREE_SCALE


def base_frame(duration: int) -> dict:
    return {"names": list(LEAVES), "matrix": {}, "highlight": (), "built": 0,
            "growth": 1.0, "caption": "", "duration_ms": duration}


def build_specs() -> "list[dict]":
    specs = []

    opening = base_frame(FIRST_HOLD_MS)
    opening["names"] = list(STEPS[0]["names"])
    opening["matrix"] = dict(STEPS[0]["matrix"])
    opening["caption"] = "One cluster per species, each a single leaf"
    specs.append(opening)

    index = 0
    while index < len(STEPS):
        step = STEPS[index]

        pick = base_frame(STEP_HOLD_MS)
        pick["names"] = list(step["names"])
        pick["matrix"] = dict(step["matrix"])
        pick["highlight"] = step["pair"]
        pick["built"] = index
        pick["caption"] = ("The closest two clusters are %s and %s, at distance %g"
                           % (step["pair"][0], step["pair"][1], step["distance"]))
        specs.append(pick)

        frame_index = 1
        while frame_index <= GROW_FRAMES:
            grow = base_frame(GROW_MS)
            grow["names"] = list(step["names"])
            grow["matrix"] = dict(step["matrix"])
            grow["highlight"] = step["pair"]
            grow["built"] = index + 1
            grow["growth"] = ease(frame_index / GROW_FRAMES)
            grow["caption"] = ("Hang a new node halfway between them, at height %g"
                               % step["age"])
            specs.append(grow)
            frame_index = frame_index + 1

        hung = base_frame(MERGE_HOLD_MS)
        hung["names"] = list(step["names"])
        hung["matrix"] = dict(step["matrix"])
        hung["highlight"] = step["pair"]
        hung["built"] = index + 1
        hung["caption"] = ("Hang a new node halfway between them, at height %g"
                           % step["age"])
        specs.append(hung)

        updated = base_frame(MERGE_HOLD_MS)
        updated["names"] = list(step["after_names"])
        updated["matrix"] = dict(step["after_matrix"])
        updated["highlight"] = (step["merged"],)
        updated["built"] = index + 1
        updated["caption"] = "Replace their rows and columns with a weighted average"
        specs.append(updated)
        index = index + 1

    final = base_frame(FINAL_HOLD_MS)
    last = STEPS[len(STEPS) - 1]
    final["names"] = list(last["after_names"])
    final["matrix"] = dict(last["after_matrix"])
    final["built"] = len(STEPS)
    final["caption"] = "One cluster left, and the tree is finished"
    specs.append(final)
    return specs


def display_name(name: str) -> str:
    if len(name) == 1:
        return name
    return "{%s}" % ",".join(name)


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)
    names = spec["names"]
    if len(names) < 2:
        # The matrix is gone once one cluster remains; name what is left instead.
        remaining = "".join(sorted(spec["names"][0]))
        axis.text(MATRIX_LEFT + 1.6 * CELL, MATRIX_TOP - 1.6 * CELL,
                  display_name(remaining), fontsize=HEADER_SIZE + 4, color=INK,
                  ha="center", va="center", fontproperties=OPTIMA_ITALIC, zorder=5)
        names = []

    column = 0
    while column < len(names):
        highlighted = names[column] in spec["highlight"]
        if highlighted:
            colour = RED
        else:
            colour = INK
        axis.text(MATRIX_LEFT + (column + 1.5) * CELL, MATRIX_TOP,
                  display_name(names[column]), fontsize=HEADER_SIZE, color=colour,
                  ha="center", va="center", fontproperties=OPTIMA_ITALIC, zorder=5)
        axis.text(MATRIX_LEFT + 0.5 * CELL, MATRIX_TOP - (column + 1) * CELL,
                  display_name(names[column]), fontsize=HEADER_SIZE, color=colour,
                  ha="center", va="center", fontproperties=OPTIMA_ITALIC, zorder=5)
        column = column + 1

    row = 0
    while row < len(names):
        column = 0
        while column < len(names):
            value = spec["matrix"][(names[row], names[column])]
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
        row = row + 1

    import matplotlib.patches as patches
    built = spec["built"]
    index = 0
    while index < built:
        step = STEPS[index]
        if index == built - 1:
            growth = spec["growth"]
        else:
            growth = 1.0
        parent_y = height_y(step["age"])
        centre_x = leaf_x(step["merged"])
        xs = []
        for child, limb in step["children"]:
            child_y = height_y(AGES[child])
            reach = child_y + (parent_y - child_y) * growth
            child_x = leaf_x(child)
            xs.append(child_x)
            axis.plot([child_x, child_x], [child_y, reach], color=INK,
                      linewidth=1.8, solid_capstyle="round", zorder=4)
            if growth > 0.98:
                # Put each limb length on the far side of its own vertical, or it
                # collides with the node age sitting under the bar.
                if child_x < centre_x:
                    offset = -0.15
                    align = "right"
                else:
                    offset = 0.15
                    align = "left"
                axis.text(child_x + offset, (child_y + parent_y) / 2, "%g" % limb,
                          fontsize=LIMB_SIZE, color=INK, ha=align, va="center",
                          fontproperties=MONO, zorder=6)
        if growth > 0.98:
            axis.plot([xs[0], xs[1]], [parent_y, parent_y], color=INK,
                      linewidth=1.8, solid_capstyle="round", zorder=4)
            axis.text(centre_x, parent_y - 0.15, "%g" % step["age"],
                      fontsize=LIMB_SIZE, color=DIM, ha="center", va="top",
                      fontproperties=MONO, zorder=6)
        index = index + 1

    for leaf in LEAVES:
        x = leaf_x(leaf)
        y = height_y(0.0)
        disc = patches.Circle((x, y), LEAF_RADIUS, facecolor=DISC,
                              edgecolor="none", zorder=5)
        axis.add_patch(disc)
        axis.text(x, y, leaf, fontsize=LEAF_SIZE, color="white", ha="center",
                  va="center", fontproperties=OPTIMA_ITALIC, zorder=6)

    if spec["caption"] != "":
        axis.text(FIGURE_WIDTH / 2, CAPTION_Y, spec["caption"], fontsize=LABEL_SIZE,
                  color=INK, ha="center", va="center", fontproperties=OPTIMA_ITALIC,
                  zorder=7)

    figure.savefig(output_path, facecolor=BACKGROUND)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: upgma.py OUTPUT.gif")
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
    directory = tempfile.mkdtemp(prefix="upgma_")
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
