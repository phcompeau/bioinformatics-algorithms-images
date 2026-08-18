"""Animated 2-break sorting on the breakpoint graph, in the book's own style.

This animates the figure the text already prints, images/Rearrangements/
2-break_series.png: P = (+a -b -c +d) sorted into Q = (+a +c +b -d) by two
2-breaks, with the red-blue cycle count climbing 2 -> 3 -> 4.

Conventions taken from that figure and from breakpoint_graph.png beside it:
blocks are letters drawn as dashed grey arrows around the circle, P's adjacencies
are red chords and Q's are blue, the two edges a 2-break cuts are starred in
yellow, and each genome is printed underneath as a signed permutation, P in red
and Q in blue.

Nothing is hand-typed except the two genomes: the node order, the block edges and
both sets of adjacency edges are derived from them, and the genome printed under
the graph is read back out of the red edges after every 2-break. Checks: the graph
is 2-regular in each colour, every 2-break raises the cycle count by exactly one,
the intermediate genome matches the one printed in the figure, and P finishes
equal to Q.

Run:  python3 example_breakpoint_graph.py OUTPUT.gif
"""

import math
import os
import sys
import tempfile

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, BLUE, DIM, FAINT, GREEN, INK, MONO,
                           MONO_BOLD, ORANGE, OPTIMA, OPTIMA_ITALIC, PURPLE, RED,
                           ease, mono_advance, new_axes)
from make_gif import assemble_transparent_gif

# The two genomes of 2-break_series.png, and the intermediate it prints.
BLOCK_NAMES = ["a", "b", "c", "d"]
P_GENOME = ["+a", "-b", "-c", "+d"]
Q_GENOME = ["+a", "+c", "+b", "-d"]
PUBLISHED_INTERMEDIATE = ["+a", "-b", "-c", "-d"]
PUBLISHED_CYCLES = [2, 3, 4]

FIGURE_WIDTH = 9.6
FIGURE_HEIGHT = 8.2
RENDER_DPI = 140
OUTPUT_WIDTH = 1344
OUTPUT_HEIGHT = 1148

CENTRE_X = 4.8
CENTRE_Y = 4.90
RADIUS = 2.35
NODE_RADIUS = 0.085
BLOCK_ARC_OUT = 0.30
BLOCK_LABEL_OUT = 0.62
BLOCK_LABEL_SIZE = 17.0
EDGE_WIDTH = 2.4
EDGE_OFFSET = 0.055
STAR_SIZE = 300.0

GENOME_Y = 1.55
CYCLES_Y = 1.05
CLOSING_Y = 0.48
GENOME_SIZE = 19.0
LABEL_SIZE = 16.0
# Node rings are coloured by which alternating cycle the node sits in, so a
# 2-break splitting a cycle shows up as one colour becoming two.
CYCLE_COLOURS = [PURPLE, ORANGE, GREEN, "#1F8A8C", "#B8860B"]

FIRST_HOLD_MS = 3400
STEP_HOLD_MS = 3000
MARK_MS = 2800
BREAK_FRAMES = 10
BREAK_MS = 120
FINAL_HOLD_MS = 5600


def signed_parts(entry: str) -> "tuple[str, int]":
    """"-b" becomes ("b", -1)."""
    if entry[0] == "-":
        return (entry[1:], -1)
    return (entry[1:], 1)


def node_walk(genome: "list[str]") -> "list[str]":
    """The block ends in the order this circular genome visits them."""
    result = []
    for entry in genome:
        name, sign = signed_parts(entry)
        if sign > 0:
            result.append(name + "t")
            result.append(name + "h")
        else:
            result.append(name + "h")
            result.append(name + "t")
    return result


def adjacencies(genome: "list[str]") -> set:
    """The edges joining consecutive blocks, closing the circle at the end."""
    walk = node_walk(genome)
    result = set()
    index = 1
    while index < len(walk):
        first = walk[index]
        second = walk[(index + 1) % len(walk)]
        result.add(normalise((first, second)))
        index = index + 2
    return result


def normalise(edge: "tuple[str, str]") -> "tuple[str, str]":
    if edge[0] <= edge[1]:
        return edge
    return (edge[1], edge[0])


def block_edges() -> "list[tuple]":
    """(tail, head, name) for each block: the arrow runs tail to head."""
    result = []
    for name in BLOCK_NAMES:
        result.append((name + "t", name + "h", name))
    return result


NODE_ORDER = node_walk(Q_GENOME)
BLUE_EDGES = adjacencies(Q_GENOME)
RED_EDGES = adjacencies(P_GENOME)
BLOCK_EDGES = block_edges()
BLOCKS = len(BLOCK_NAMES)


def count_cycles(red: set, blue: set) -> int:
    """Alternating red-blue cycles; every node carries exactly one of each."""
    red_at = {}
    blue_at = {}
    for first, second in red:
        red_at[first] = second
        red_at[second] = first
    for first, second in blue:
        blue_at[first] = second
        blue_at[second] = first
    seen = set()
    cycles = 0
    for start in NODE_ORDER:
        if start in seen:
            continue
        cycles = cycles + 1
        node = start
        use_red = True
        while node not in seen:
            seen.add(node)
            if use_red:
                node = red_at[node]
            else:
                node = blue_at[node]
            use_red = not use_red
    return cycles


def cycle_membership(red: set, blue: set) -> dict:
    """Which alternating cycle each node belongs to."""
    red_at = {}
    blue_at = {}
    for first, second in red:
        red_at[first] = second
        red_at[second] = first
    for first, second in blue:
        blue_at[first] = second
        blue_at[second] = first
    result = {}
    index = 0
    for start in NODE_ORDER:
        if start in result:
            continue
        node = start
        use_red = True
        while node not in result:
            result[node] = index
            if use_red:
                node = red_at[node]
            else:
                node = blue_at[node]
            use_red = not use_red
        index = index + 1
    return result


def genome_of(red: set) -> "list[str]":
    """Read the genome back out of its adjacency edges.

    Walk a block tail to head, cross the adjacency waiting there, and carry on
    until the walk closes. This is what lets the printed permutation under the
    graph change as the red edges change, instead of being typed in.
    """
    partner = {}
    for first, second in red:
        partner[first] = second
        partner[second] = first
    result = []
    used = set()
    current = BLOCK_NAMES[0] + "t"
    while True:
        name = current[:len(current) - 1]
        end = current[len(current) - 1]
        if name in used:
            break
        used.add(name)
        if end == "t":
            result.append("+" + name)
            current = partner[name + "h"]
        else:
            result.append("-" + name)
            current = partner[name + "t"]
    return result


def plan_two_breaks() -> "list[dict]":
    """The 2-breaks the figure performs, derived from the genomes it prints.

    A greedy rule picks a legal sequence too, but not this one: there are several
    two-step sortings of P and the figure shows one of them. So the sequence of
    genomes is taken from the figure, and each step's cut and rejoin is derived by
    differencing consecutive adjacency sets. verify() then checks that every step
    really is a 2-break: two edges out, two edges in, the same four block ends, and
    one more cycle.
    """
    sequence = [P_GENOME, PUBLISHED_INTERMEDIATE, Q_GENOME]
    steps = []
    index = 1
    while index < len(sequence):
        red = adjacencies(sequence[index - 1])
        new_red = adjacencies(sequence[index])
        removed = []
        for edge in sorted(red - new_red):
            removed.append(edge)
        added = []
        for edge in sorted(new_red - red):
            added.append(edge)
        steps.append({"removed": removed, "added": added,
                      "before": count_cycles(red, BLUE_EDGES),
                      "after": count_cycles(new_red, BLUE_EDGES),
                      "red_before": set(red), "red_after": set(new_red),
                      "cycles_before": cycle_membership(red, BLUE_EDGES),
                      "cycles_after": cycle_membership(new_red, BLUE_EDGES),
                      "genome_before": list(sequence[index - 1]),
                      "genome_after": list(sequence[index])})
        index = index + 1
    return steps


STEPS = plan_two_breaks()
START_CYCLES = count_cycles(RED_EDGES, BLUE_EDGES)


def node_angle(name: str) -> float:
    """Nodes sit evenly around the circle in the order Q visits them."""
    index = NODE_ORDER.index(name)
    return math.radians(202.5 - index * (360.0 / len(NODE_ORDER)))


def node_xy(name: str) -> "tuple[float, float]":
    angle = node_angle(name)
    return (CENTRE_X + RADIUS * math.cos(angle), CENTRE_Y + RADIUS * math.sin(angle))


def trimmed(first: "tuple[float, float]",
            second: "tuple[float, float]") -> "tuple":
    """Stop each edge just short of the two dots it joins."""
    length = math.hypot(second[0] - first[0], second[1] - first[1])
    unit_x = (second[0] - first[0]) / length
    unit_y = (second[1] - first[1]) / length
    gap = NODE_RADIUS + 0.03
    return ((first[0] + unit_x * gap, first[1] + unit_y * gap),
            (second[0] - unit_x * gap, second[1] - unit_y * gap))


def written(genome: "list[str]") -> str:
    return "(" + " ".join(genome).replace("-", "−") + ")"


def verify() -> "list[str]":
    lines = []

    for node in NODE_ORDER:
        red_count = 0
        blue_count = 0
        for edge in RED_EDGES:
            if node in edge:
                red_count = red_count + 1
        for edge in BLUE_EDGES:
            if node in edge:
                blue_count = blue_count + 1
        assert red_count == 1 and blue_count == 1, (
            "%s carries %d red and %d blue edges" % (node, red_count, blue_count))
    lines.append("each of the %d block ends carries exactly one red and one blue edge"
                 % len(NODE_ORDER))

    assert genome_of(RED_EDGES) == P_GENOME, (
        "reading the red edges back gives %s, not P" % genome_of(RED_EDGES))
    assert genome_of(BLUE_EDGES) == Q_GENOME, (
        "reading the blue edges back gives %s, not Q" % genome_of(BLUE_EDGES))
    lines.append("both edge sets read back as the genomes they came from: P = %s, "
                 "Q = %s" % (written(P_GENOME), written(Q_GENOME)))

    assert START_CYCLES == PUBLISHED_CYCLES[0], (
        "the figure prints Cycles(P, Q) = %d but this graph has %d"
        % (PUBLISHED_CYCLES[0], START_CYCLES))
    counts = [START_CYCLES]
    for step in STEPS:
        counts.append(step["after"])
    assert counts == PUBLISHED_CYCLES, (
        "the cycle counts run %s but the figure prints %s" % (counts, PUBLISHED_CYCLES))
    lines.append("the cycle count runs %s, exactly as 2-break_series.png prints it"
                 % " -> ".join(str(value) for value in counts))

    assert len(STEPS) == 2, "the figure sorts P in two 2-breaks, got %d" % len(STEPS)
    assert STEPS[0]["genome_after"] == PUBLISHED_INTERMEDIATE, (
        "after the first 2-break this gives %s but the figure prints %s"
        % (STEPS[0]["genome_after"], PUBLISHED_INTERMEDIATE))
    assert STEPS[1]["genome_after"] == Q_GENOME, "the second 2-break must land on Q"
    lines.append("the intermediate genome is %s and the last is Q, matching the figure"
                 % written(PUBLISHED_INTERMEDIATE))

    for step in STEPS:
        assert len(step["removed"]) == 2 and len(step["added"]) == 2, (
            "a step cut %d and added %d edges, so it is not a 2-break"
            % (len(step["removed"]), len(step["added"])))
        assert step["after"] == step["before"] + 1, "a 2-break added no cycle"
        ends_removed = set()
        for edge in step["removed"]:
            ends_removed.add(edge[0])
            ends_removed.add(edge[1])
        ends_added = set()
        for edge in step["added"]:
            ends_added.add(edge[0])
            ends_added.add(edge[1])
        assert ends_removed == ends_added, "a 2-break moved a block end"
    lines.append("each 2-break rejoins the same four block ends and adds one cycle")

    assert STEPS[len(STEPS) - 1]["red_after"] == BLUE_EDGES, "P must finish as Q"
    assert count_cycles(BLUE_EDGES, BLUE_EDGES) == BLOCKS, (
        "the trivial graph should have one cycle per block")
    lines.append("P finishes identical to Q, with all %d cycles trivial" % BLOCKS)

    lowest = CENTRE_Y
    for name in NODE_ORDER:
        if node_xy(name)[1] < lowest:
            lowest = node_xy(name)[1]
    assert lowest - BLOCK_LABEL_OUT - 0.2 > GENOME_Y, (
        "the circle collides with the printed genomes")
    assert CENTRE_Y + RADIUS + BLOCK_LABEL_OUT + 0.2 < FIGURE_HEIGHT, (
        "the circle runs off the top")
    assert CENTRE_X + RADIUS + BLOCK_LABEL_OUT + 0.2 < FIGURE_WIDTH, (
        "the circle runs off the right")
    assert CLOSING_Y + 0.25 < CYCLES_Y, "the closing line collides with the count"
    lines.append("circle of radius %.2f in sits clear of the printed genomes"
                 % RADIUS)
    return lines


def base_frame(duration: int) -> dict:
    return {"red": set(), "fading": [], "rising": [], "progress": 0.0,
            "cycles": 0, "cycle_of": {}, "marked": [], "genome": [],
            "closing": "", "duration_ms": duration}


def build_specs() -> "list[dict]":
    specs = []

    only_q = base_frame(FIRST_HOLD_MS)
    specs.append(only_q)

    arrive = base_frame(STEP_HOLD_MS)
    arrive["red"] = set(RED_EDGES)
    arrive["cycles"] = START_CYCLES
    arrive["genome"] = list(P_GENOME)
    specs.append(arrive)

    both = base_frame(STEP_HOLD_MS)
    both["red"] = set(RED_EDGES)
    both["cycles"] = START_CYCLES
    both["cycle_of"] = cycle_membership(RED_EDGES, BLUE_EDGES)
    both["genome"] = list(P_GENOME)
    specs.append(both)

    index = 0
    while index < len(STEPS):
        step = STEPS[index]

        mark = base_frame(MARK_MS)
        mark["red"] = set(step["red_before"])
        mark["cycles"] = step["before"]
        mark["cycle_of"] = dict(step["cycles_before"])
        mark["marked"] = list(step["removed"])
        mark["genome"] = list(step["genome_before"])
        specs.append(mark)

        frame_index = 1
        while frame_index <= BREAK_FRAMES:
            moving = base_frame(BREAK_MS)
            moving["red"] = set(step["red_before"]) - set(step["removed"])
            moving["fading"] = list(step["removed"])
            moving["rising"] = list(step["added"])
            moving["progress"] = ease(frame_index / BREAK_FRAMES)
            moving["cycles"] = step["before"]
            moving["cycle_of"] = dict(step["cycles_before"])
            moving["genome"] = list(step["genome_before"])
            specs.append(moving)
            frame_index = frame_index + 1

        settled = base_frame(STEP_HOLD_MS)
        settled["red"] = set(step["red_after"])
        settled["cycles"] = step["after"]
        settled["cycle_of"] = dict(step["cycles_after"])
        settled["genome"] = list(step["genome_after"])
        specs.append(settled)
        index = index + 1

    final = base_frame(FINAL_HOLD_MS)
    final["red"] = set(STEPS[len(STEPS) - 1]["red_after"])
    final["cycles"] = BLOCKS
    final["cycle_of"] = cycle_membership(BLUE_EDGES, BLUE_EDGES)
    final["genome"] = list(Q_GENOME)
    final["closing"] = "%d 2-breaks turned P into Q" % len(STEPS)
    specs.append(final)
    return specs


def draw_edge(axis: "plt.Axes", edge: "tuple[str, str]", colour: str,
              alpha: float, offset: float) -> None:
    """One adjacency edge, shifted to its own side so red and blue both show."""
    if alpha <= 0.02:
        return
    first, second = trimmed(node_xy(edge[0]), node_xy(edge[1]))
    length = math.hypot(second[0] - first[0], second[1] - first[1])
    shift_x = -(second[1] - first[1]) / length * offset
    shift_y = (second[0] - first[0]) / length * offset
    axis.plot([first[0] + shift_x, second[0] + shift_x],
              [first[1] + shift_y, second[1] + shift_y], color=colour,
              linewidth=EDGE_WIDTH, solid_capstyle="round", zorder=4, alpha=alpha)


def draw_blocks(axis: "plt.Axes") -> None:
    """Each block as a dashed grey arrow bowing outside the circle, with its letter."""
    for tail, head, name in BLOCK_EDGES:
        start = node_xy(tail)
        end = node_xy(head)
        middle_angle = (node_angle(tail) + node_angle(head)) / 2
        if abs(node_angle(tail) - node_angle(head)) > math.pi:
            middle_angle = middle_angle + math.pi
        arrow = patches.FancyArrowPatch(
            start, end, connectionstyle="arc3,rad=-0.22",
            arrowstyle="-|>", mutation_scale=16, linewidth=2.0, linestyle=(0, (4, 3)),
            color=DIM, shrinkA=6.0, shrinkB=6.0, zorder=3)
        axis.add_patch(arrow)
        axis.text(CENTRE_X + (RADIUS + BLOCK_LABEL_OUT) * math.cos(middle_angle),
                  CENTRE_Y + (RADIUS + BLOCK_LABEL_OUT) * math.sin(middle_angle),
                  name, fontsize=BLOCK_LABEL_SIZE, color=DIM, ha="center",
                  va="center", fontproperties=OPTIMA_ITALIC, zorder=6)


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)

    draw_blocks(axis)
    for edge in BLUE_EDGES:
        draw_edge(axis, edge, BLUE, 1.0, -EDGE_OFFSET)
    for edge in spec["red"]:
        draw_edge(axis, edge, RED, 1.0, EDGE_OFFSET)
    for edge in spec["fading"]:
        draw_edge(axis, edge, RED, 1.0 - spec["progress"], EDGE_OFFSET)
    for edge in spec["rising"]:
        draw_edge(axis, edge, RED, spec["progress"], EDGE_OFFSET)

    # The book stars the two edges a 2-break is about to cut.
    for edge in spec["marked"]:
        first = node_xy(edge[0])
        second = node_xy(edge[1])
        axis.scatter([(first[0] + second[0]) / 2], [(first[1] + second[1]) / 2],
                     s=STAR_SIZE, marker="*", facecolor="#FFE800",
                     edgecolor="#8A7300", linewidth=1.0, zorder=7)

    for name in NODE_ORDER:
        x, y = node_xy(name)
        if name in spec["cycle_of"]:
            ring = CYCLE_COLOURS[spec["cycle_of"][name] % len(CYCLE_COLOURS)]
        else:
            ring = INK
        halo = patches.Circle((x, y), NODE_RADIUS + 0.045, facecolor=ring,
                              edgecolor="none", zorder=5)
        axis.add_patch(halo)
        dot = patches.Circle((x, y), NODE_RADIUS, facecolor=INK, edgecolor="none",
                             zorder=6)
        axis.add_patch(dot)

    advance = mono_advance(figure, GENOME_SIZE)
    if len(spec["genome"]) > 0:
        pieces = [("P = ", RED), (written(spec["genome"]), RED),
                  ("     Q = ", BLUE), (written(Q_GENOME), BLUE)]
        total = 0
        for text, colour in pieces:
            total = total + len(text)
        x = CENTRE_X - total * advance / 2
        for text, colour in pieces:
            axis.text(x, GENOME_Y, text, fontsize=GENOME_SIZE, color=colour,
                      ha="left", va="center", fontproperties=MONO_BOLD, zorder=7)
            x = x + len(text) * advance

    if spec["cycles"] > 0:
        axis.text(CENTRE_X, CYCLES_Y, "Cycles(P, Q) = %d" % spec["cycles"],
                  fontsize=GENOME_SIZE, color=INK, ha="center", va="center",
                  fontproperties=MONO_BOLD, zorder=7)

    if spec["closing"] != "":
        axis.text(CENTRE_X, CLOSING_Y, spec["closing"], fontsize=LABEL_SIZE,
                  color=INK, ha="center", va="center", fontproperties=OPTIMA_ITALIC,
                  zorder=7)

    figure.savefig(output_path, transparent=True)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: example_breakpoint_graph.py OUTPUT.gif")
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
    directory = tempfile.mkdtemp(prefix="breakpoint_graph_")
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
