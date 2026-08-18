"""Animated UPGMA clustering, in the 02-180 lecture style.

Two animations, one per worked example in Evolutionary_Trees.pptx:

  basic - slides 110-118. Four species, every cluster the same size, so the
          average distance is a plain average.
  quiz  - slides 121-129, the "Quick UPGMA Quiz". Here the clusters end up
          different sizes, the plain average gives the wrong answer (17 instead
          of 16), and the animation shows both so the weighting has a point.

Each run shows the arithmetic for every new matrix entry: which two old
distances are being combined, what each cluster's size is, and the result.

Both matrices are read from the deck's tables. The node ages and limb lengths
are asserted against the numbers printed on the slides, the tree is checked for
the ultrametric property UPGMA guarantees, and every averaged entry is checked
against the direct average over all pairs of leaves across the two clusters,
which is what the weighted recurrence is supposed to compute.

Run:  python3 example_upgma.py basic OUTPUT.gif
      python3 example_upgma.py quiz OUTPUT.gif
"""

import os
import sys
import tempfile

import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, DIM, DISC, FAINT, GREEN, INK, MONO,
                           MONO_BOLD, RED, OPTIMA, OPTIMA_ITALIC, ease,
                           mono_advance, new_axes)
from make_gif import assemble_transparent_gif

# Both matrices are transcribed from the tables in Evolutionary_Trees.pptx:
# slide 110 for the walkthrough, slide 121 for the quiz. PUBLISHED lists every
# age and limb length printed beside the finished tree.
# The book prints node ages in this red, and names the four species v1..v4
# (sampled and read off images/Evolution/UPGMA.png, the figure these sit beside).
# Cluster names are built by concatenating the single-character keys, so the keys
# stay i j k l internally and only the labels change.
BOOK_RED = "#C80000"
LABEL_OF = {"i": "v1", "j": "v2", "k": "v3", "l": "v4"}

EXAMPLES = {
    "basic": {
        "leaves": ["i", "j", "k", "l"],
        "matrix": [[0, 3, 4, 3],
                   [3, 0, 4, 5],
                   [4, 4, 0, 2],
                   [3, 5, 2, 0]],
        "published": [0.5, 1, 1, 1, 1, 1.5, 1.5, 1.5, 2],
        "source": "slides 110-118",
    },
    "quiz": {
        "leaves": ["i", "j", "k", "l"],
        "matrix": [[0, 20, 9, 11],
                   [20, 0, 17, 11],
                   [9, 17, 0, 8],
                   [11, 11, 8, 0]],
        "published": [1, 3, 4, 4, 4, 5, 5, 8, 8],
        "source": "slides 121-129",
    },
}

if len(sys.argv) > 1 and sys.argv[1] in EXAMPLES:
    WHICH = sys.argv[1]
else:
    WHICH = "basic"

LEAVES = EXAMPLES[WHICH]["leaves"]
DISTANCES = EXAMPLES[WHICH]["matrix"]
PUBLISHED_NUMBERS = EXAMPLES[WHICH]["published"]
SOURCE = EXAMPLES[WHICH]["source"]

FIGURE_WIDTH = 10.0
FIGURE_HEIGHT = 6.05
RENDER_DPI = 140
OUTPUT_WIDTH = 1400
OUTPUT_HEIGHT = 847

CELL = 0.62
MATRIX_LEFT = 0.75
MATRIX_TOP = 4.40
HEADER_SIZE = 13.0
VALUE_SIZE = 14.0

TREE_LEFT = 5.85
TREE_LEAF_GAP = 1.05
TREE_BASE = 1.40
LEAF_RADIUS = 0.17
LEAF_SIZE = 12.0
LIMB_SIZE = 12.0

CAPTION_Y = 5.63
LABEL_SIZE = 15.0
# The averaging arithmetic sits along the bottom, where it has the full width.
WORK_Y = 0.74
TRAP_Y = 0.34
WORK_SIZE = 15.0

STEP_HOLD_MS = 2600
GROW_FRAMES = 9
GROW_MS = 110
MERGE_HOLD_MS = 2200
WORK_MS = 2800
TRAP_MS = 4200
FIRST_HOLD_MS = 3000
FINAL_HOLD_MS = 5200


def leaves_of(name: str) -> "list[str]":
    """A cluster's name is the concatenation of its leaves."""
    result = []
    for character in name:
        result.append(character)
    return result


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
                "left_size": sizes[left],
                "right_size": sizes[right],
                "children": [(left, age - ages[left]), (right, age - ages[right])]}
        new_names = []
        for name in names:
            if name != left and name != right:
                new_names.append(name)
        new_names.append(merged)
        sizes[merged] = sizes[left] + sizes[right]
        ages[merged] = age
        # One work item per new matrix entry: the two old distances being
        # combined, the two cluster sizes doing the weighting, and the answer.
        work = []
        for name in new_names:
            if name == merged:
                matrix[(merged, merged)] = 0.0
                continue
            from_left = matrix[(left, name)]
            from_right = matrix[(right, name)]
            weighted = (sizes[left] * from_left + sizes[right] * from_right)
            value = weighted / (sizes[left] + sizes[right])
            matrix[(merged, name)] = value
            matrix[(name, merged)] = value
            work.append({"other": name, "from_left": from_left,
                         "from_right": from_right, "value": value,
                         "plain": (from_left + from_right) / 2.0})
        step["work"] = work
        names = new_names
        step["after_names"] = list(names)
        step["after_matrix"] = dict(matrix)
        record.append(step)
    return record


STEPS = run_upgma()


def cluster_sizes() -> dict:
    result = {}
    for name in LEAVES:
        result[name] = 1
    for step in STEPS:
        result[step["merged"]] = step["left_size"] + step["right_size"]
    return result


SIZES = cluster_sizes()


def node_ages() -> dict:
    result = {}
    for name in LEAVES:
        result[name] = 0.0
    for step in STEPS:
        result[step["merged"]] = step["age"]
    return result


AGES = node_ages()
ROOT = STEPS[len(STEPS) - 1]["merged"]


def leaf_slot() -> dict:
    """Left-to-right leaf order taken from the merge tree, not from the input.

    Ordering the leaves by the order they get merged keeps every cluster
    contiguous, so no limb ever has to cross another one.
    """
    order = leaves_of(ROOT)
    result = {}
    index = 0
    while index < len(order):
        result[order[index]] = float(index)
        index = index + 1
    return result


SLOT = leaf_slot()


def cluster_slot(name: str) -> float:
    total = 0.0
    count = 0
    for character in leaves_of(name):
        total = total + SLOT[character]
        count = count + 1
    return total / count


def tree_scale() -> float:
    """Fit the tallest node into the space above the leaves."""
    return (CAPTION_Y - 0.55 - TREE_BASE) / AGES[ROOT]


TREE_SCALE = tree_scale()


def direct_average(first: str, second: str) -> float:
    """Mean distance over every pair of leaves, one from each cluster.

    This is the definition UPGMA's weighted recurrence is a shortcut for, so it
    is the right thing to check the recurrence against.
    """
    total = 0.0
    count = 0
    for one in leaves_of(first):
        for other in leaves_of(second):
            total = total + DISTANCES[LEAVES.index(one)][LEAVES.index(other)]
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
    lines.append("%s matrix from %s is symmetric with a zero diagonal"
                 % (WHICH, SOURCE))

    assert len(STEPS) == size - 1, "n-1 merges for n leaves"
    lines.append("UPGMA performed %d merges for %d species" % (len(STEPS), size))

    numbers = []
    for step in STEPS:
        numbers.append(step["age"])
        for child, limb in step["children"]:
            numbers.append(limb)
    numbers.sort()
    assert numbers == sorted(PUBLISHED_NUMBERS), (
        "computed %s but the slides print %s" % (numbers, sorted(PUBLISHED_NUMBERS)))
    lines.append("node ages and limb lengths are %s, matching the slides"
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

    # The weighted recurrence must agree with the average over all leaf pairs.
    # This is the check that would have caught an unweighted average.
    disagreements = 0
    for step in STEPS:
        for item in step["work"]:
            expected = direct_average(step["merged"], item["other"])
            assert abs(item["value"] - expected) < 1e-9, (
                "D(%s, %s) came out %g but averaging all leaf pairs gives %g"
                % (step["merged"], item["other"], item["value"], expected))
            if abs(item["plain"] - expected) > 1e-9:
                disagreements = disagreements + 1
    lines.append("every averaged entry matches the mean over all leaf pairs")
    if disagreements > 0:
        lines.append("%d of those entries would be WRONG without the size weights, "
                     "which is what this example is for" % disagreements)
    else:
        lines.append("all clusters stay the same size here, so no entry needs the "
                     "weights yet")

    for leaf in LEAVES:
        depth = 0.0
        name = leaf
        while name != ROOT:
            parent = ""
            for step in STEPS:
                if name in step["pair"]:
                    parent = step["merged"]
                    for child, limb in step["children"]:
                        if child == name:
                            depth = depth + limb
            assert parent != "", "leaf %s has no path to the root" % leaf
            name = parent
        assert abs(depth - AGES[ROOT]) < 1e-9, (
            "leaf %s sits %g from the root, not %g" % (leaf, depth, AGES[ROOT]))
    lines.append("all %d leaves are %g from the root: the tree is ultrametric"
                 % (size, AGES[ROOT]))

    top = TREE_BASE + AGES[ROOT] * TREE_SCALE
    assert top + 0.3 < CAPTION_Y, "tree runs into the caption"
    width = MATRIX_LEFT + (size + 1) * CELL
    assert width < TREE_LEFT - 0.3, "matrix collides with the tree"
    lowest = MATRIX_TOP - size * CELL
    assert lowest > WORK_Y + 0.3, "matrix collides with the arithmetic"
    assert TREE_BASE - LEAF_RADIUS > WORK_Y + 0.25, "tree collides with the arithmetic"
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
            "growth": 1.0, "caption": "", "work": None, "cells": (),
            "trap": False, "duration_ms": duration}


def display_name(name: str) -> str:
    labels = []
    for character in leaves_of(name):
        labels.append(LABEL_OF[character])
    if len(labels) == 1:
        return labels[0]
    return "{%s}" % ",".join(labels)


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
                           % (display_name(step["pair"][0]),
                              display_name(step["pair"][1]), step["distance"]))
        specs.append(pick)

        frame_index = 1
        while frame_index <= GROW_FRAMES:
            grow = base_frame(GROW_MS)
            grow["names"] = list(step["names"])
            grow["matrix"] = dict(step["matrix"])
            grow["highlight"] = step["pair"]
            grow["built"] = index + 1
            grow["growth"] = ease(frame_index / GROW_FRAMES)
            grow["caption"] = ("Their common ancestor goes in at height %g, half "
                               "their distance" % step["age"])
            specs.append(grow)
            frame_index = frame_index + 1

        hung = base_frame(MERGE_HOLD_MS)
        hung["names"] = list(step["names"])
        hung["matrix"] = dict(step["matrix"])
        hung["highlight"] = step["pair"]
        hung["built"] = index + 1
        hung["caption"] = ("Their common ancestor goes in at height %g, half "
                           "their distance" % step["age"])
        specs.append(hung)

        # Phillip: the averaging was invisible. Now every new entry is worked out
        # on screen, with the two old distances it comes from lit up in the old
        # matrix and the cluster sizes shown doing the weighting.
        for item in step["work"]:
            work = base_frame(WORK_MS)
            work["names"] = list(step["names"])
            work["matrix"] = dict(step["matrix"])
            work["built"] = index + 1
            work["highlight"] = step["pair"]
            work["cells"] = ((step["pair"][0], item["other"]),
                             (step["pair"][1], item["other"]))
            work["work"] = dict(item)
            work["work"]["merged"] = step["merged"]
            work["work"]["left"] = step["pair"][0]
            work["work"]["right"] = step["pair"][1]
            work["work"]["left_size"] = step["left_size"]
            work["work"]["right_size"] = step["right_size"]
            work["caption"] = ("Average the two old distances, weighted by how many "
                               "species each cluster holds")
            specs.append(work)

            if abs(item["plain"] - item["value"]) > 1e-9:
                trap = base_frame(TRAP_MS)
                trap["names"] = list(step["names"])
                trap["matrix"] = dict(step["matrix"])
                trap["built"] = index + 1
                trap["highlight"] = step["pair"]
                trap["cells"] = work["cells"]
                trap["work"] = dict(work["work"])
                trap["trap"] = True
                trap["caption"] = ("Averaging the two entries instead would give %g, "
                                   "which is wrong" % item["plain"])
                specs.append(trap)

        updated = base_frame(MERGE_HOLD_MS)
        updated["names"] = list(step["after_names"])
        updated["matrix"] = dict(step["after_matrix"])
        updated["highlight"] = (step["merged"],)
        updated["built"] = index + 1
        updated["caption"] = "The merged cluster takes one row and one column"
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


def draw_work(axis: "plt.Axes", figure: "plt.Figure", work: dict,
              trap: bool) -> None:
    """The weighted average spelled out, in one monospace line.

    Written as pieces so the two old distances can carry the same colours they
    have in the matrix above, and the answer can stand out from the working.
    """
    pieces = [("D(%s, %s)  =  " % (display_name(work["merged"]),
                                   display_name(work["other"])), INK),
              ("%d" % work["left_size"], DIM),
              (" · ", DIM),
              ("%g" % work["from_left"], RED),
              ("  +  ", DIM),
              ("%d" % work["right_size"], DIM),
              (" · ", DIM),
              ("%g" % work["from_right"], RED),
              ("  /  ", DIM),
              ("(%d + %d)" % (work["left_size"], work["right_size"]), DIM),
              ("   =   ", DIM),
              ("%g" % work["value"], GREEN)]
    total = 0
    for piece, colour in pieces:
        total = total + len(piece)
    advance = mono_advance(figure, WORK_SIZE)
    x = FIGURE_WIDTH / 2 - total * advance / 2
    for piece, colour in pieces:
        axis.text(x, WORK_Y, piece, fontsize=WORK_SIZE, color=colour, ha="left",
                  va="center", fontproperties=MONO_BOLD, zorder=7)
        x = x + len(piece) * advance

    if trap:
        plain = "(%g + %g) / 2  =  %g" % (work["from_left"], work["from_right"],
                                          work["plain"])
        axis.text(FIGURE_WIDTH / 2, TRAP_Y, plain, fontsize=WORK_SIZE, color=RED,
                  ha="center", va="center", fontproperties=MONO_BOLD, zorder=7)


def draw_frame(spec: dict, output_path: str) -> None:
    import matplotlib.patches as patches
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)
    names = spec["names"]
    if len(names) < 2:
        # The matrix is gone once one cluster remains; name what is left instead.
        axis.text(MATRIX_LEFT + 1.6 * CELL, MATRIX_TOP - 1.6 * CELL,
                  display_name(names[0]), fontsize=HEADER_SIZE + 4, color=INK,
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
            in_work = ((names[row], names[column]) in spec["cells"]
                       or (names[column], names[row]) in spec["cells"])
            pair_hit = (names[row] in spec["highlight"]
                        and names[column] in spec["highlight"]
                        and names[row] != names[column])
            if in_work:
                colour = RED
                font = MONO_BOLD
            elif pair_hit:
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
                      fontsize=LIMB_SIZE, color=BOOK_RED, ha="center", va="top",
                      fontproperties=MONO, zorder=6)
        index = index + 1

    for leaf in LEAVES:
        x = leaf_x(leaf)
        y = height_y(0.0)
        disc = patches.Circle((x, y), LEAF_RADIUS, facecolor=DISC,
                              edgecolor="none", zorder=5)
        axis.add_patch(disc)
        axis.text(x, y, LABEL_OF[leaf], fontsize=LEAF_SIZE - 1, color="white",
                  ha="center", va="center", fontproperties=OPTIMA_ITALIC, zorder=6)

    if spec["work"] is not None:
        draw_work(axis, figure, spec["work"], spec["trap"])

    if spec["caption"] != "":
        axis.text(FIGURE_WIDTH / 2, CAPTION_Y, spec["caption"], fontsize=LABEL_SIZE,
                  color=INK, ha="center", va="center", fontproperties=OPTIMA_ITALIC,
                  zorder=7)

    figure.savefig(output_path, transparent=True)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: example_upgma.py basic|quiz OUTPUT.gif")
        return
    output_path = sys.argv[2]

    print("Structural checks (%s):" % WHICH)
    for line in verify():
        print("  ok: " + line)

    specs = build_specs()
    durations = []
    for spec in specs:
        durations.append(spec["duration_ms"])
    print("Rendering %d frames at %dx%d, %.1f s of playback..."
          % (len(specs), OUTPUT_WIDTH, OUTPUT_HEIGHT, sum(durations) / 1000.0))
    directory = tempfile.mkdtemp(prefix="upgma_" + WHICH + "_")
    frame_paths = []
    index = 0
    while index < len(specs):
        path = os.path.join(directory, "frame_%04d.png" % index)
        draw_frame(specs[index], path)
        frame_paths.append(path)
        index = index + 1

    assemble_transparent_gif(frame_paths, output_path, width=OUTPUT_WIDTH,
                             height=OUTPUT_HEIGHT, frame_durations=durations)
    print("Saved " + output_path)


if __name__ == "__main__":
    main()
