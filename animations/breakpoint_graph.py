"""Animated 2-break sorting on the breakpoint graph, in the lecture style.

Follows the Fragile Genomes deck: build the breakpoint graph by drawing genome P's
adjacencies in red and genome Q's in blue on the same block ends. Every node has
one red and one blue edge, so the graph splits into alternating cycles. Each
2-break that splits a cycle brings P one step closer to Q, and the number of
2-breaks needed is blocks minus cycles.

Everything is asserted: the graph is a perfect alternating 2-regular graph, each
2-break raises the cycle count by exactly one, the number of 2-breaks equals
blocks minus the starting cycle count, and P's edges finish identical to Q's.

Run:  python3 example_breakpoint_graph.py OUTPUT.gif
"""

import math
import os
import sys
import tempfile

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, BLUE, DIM, FAINT, GREEN, INK, MONO,
                           MONO_BOLD, ORANGE, OPTIMA_ITALIC, PURPLE, RED, ease,
                           new_axes)
from make_gif import assemble_transparent_gif

BLOCKS = 4
# Node order around the circle: block tails and heads as Q lays them out.
NODES = ["1t", "1h", "2t", "2h", "3t", "3h", "4t", "4h"]
# P = (+1 -2 -3 +4), Q = (+1 +2 +3 +4), both circular.
RED_EDGES = [("1h", "2h"), ("2t", "3h"), ("3t", "4t"), ("4h", "1t")]
BLUE_EDGES = [("1h", "2t"), ("2h", "3t"), ("3h", "4t"), ("4h", "1t")]

FIGURE_WIDTH = 9.4
FIGURE_HEIGHT = 6.8
RENDER_DPI = 150
OUTPUT_WIDTH = 1410
OUTPUT_HEIGHT = 1020

CENTRE_X = 4.7
CENTRE_Y = 3.15
RADIUS = 1.95
NODE_RADIUS = 0.155
NODE_SIZE = 10.0
BLOCK_SIZE = 14.0
EDGE_WIDTH = 2.2
EDGE_OFFSET = 0.055
COUNT_Y = 0.42
COUNT_SIZE = 22.0
LABEL_SIZE = 15.0
CAPTION_Y = 6.50
LEGEND_Y = 6.08
LEGEND_SIZE = 13.0
BLOCK_LABEL_OUT = 0.36
# Phillip could not see what was changing or why. Every node now wears the colour
# of the cycle it belongs to, so a 2-break splitting one cycle into two is a
# colour splitting in two, and each step says what it is doing.
CYCLE_COLOURS = [PURPLE, ORANGE, GREEN, "#1F8A8C", "#B8860B"]

MARK_MS = 2800
STEP_HOLD_MS = 3000
BREAK_FRAMES = 10
BREAK_MS = 120
FIRST_HOLD_MS = 3200
FINAL_HOLD_MS = 5600


def normalise(edge: "tuple[str, str]") -> "tuple[str, str]":
    if edge[0] <= edge[1]:
        return edge
    return (edge[1], edge[0])


def edge_set(edges: "list") -> set:
    result = set()
    for edge in edges:
        result.add(normalise(edge))
    return result


def count_cycles(red: set, blue: set) -> int:
    """Alternating red-blue cycles; every node has exactly one of each."""
    red_at = {}
    blue_at = {}
    for a, b in red:
        red_at[a] = b
        red_at[b] = a
    for a, b in blue:
        blue_at[a] = b
        blue_at[b] = a
    seen = set()
    cycles = 0
    for node in NODES:
        if node in seen:
            continue
        cycles = cycles + 1
        walker = node
        use_red = True
        while True:
            seen.add(walker)
            if use_red:
                walker = red_at[walker]
            else:
                walker = blue_at[walker]
            use_red = not use_red
            if walker == node and use_red:
                break
    return cycles


def cycle_membership(red: set, blue: set) -> dict:
    """Which alternating cycle each node belongs to.

    Every node carries exactly one red and one blue edge, so following red, blue,
    red, blue from any node walks a cycle and comes back. Colouring nodes by cycle
    is what makes a 2-break visible: one colour becomes two.
    """
    red_at = {}
    blue_at = {}
    for a, b in red:
        red_at[a] = b
        red_at[b] = a
    for a, b in blue:
        blue_at[a] = b
        blue_at[b] = a
    result = {}
    index = 0
    for start in NODES:
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


def plan_two_breaks() -> "list[dict]":
    """Repeatedly split a non-trivial cycle by pulling one blue edge into red."""
    red = edge_set(RED_EDGES)
    blue = edge_set(BLUE_EDGES)
    steps = []
    while red != blue:
        red_at = {}
        for a, b in red:
            red_at[a] = b
            red_at[b] = a
        chosen = None
        for a, b in sorted(blue):
            if normalise((a, b)) in red:
                continue
            chosen = (a, b)
            break
        first, second = chosen
        other_first = red_at[first]
        other_second = red_at[second]
        removed = [normalise((first, other_first)), normalise((second, other_second))]
        added = [normalise((first, second)), normalise((other_first, other_second))]
        before = count_cycles(red, blue)
        new_red = set(red)
        for edge in removed:
            new_red.discard(edge)
        for edge in added:
            new_red.add(edge)
        after = count_cycles(new_red, blue)
        steps.append({"removed": list(removed), "added": list(added),
                      "before": before, "after": after,
                      "red_before": set(red), "red_after": set(new_red),
                      "cycles_before": cycle_membership(red, blue),
                      "cycles_after": cycle_membership(new_red, blue)})
        red = new_red
    return steps


STEPS = plan_two_breaks()
START_CYCLES = count_cycles(edge_set(RED_EDGES), edge_set(BLUE_EDGES))


def verify() -> "list[str]":
    lines = []
    red = edge_set(RED_EDGES)
    blue = edge_set(BLUE_EDGES)

    assert len(NODES) == 2 * BLOCKS, "two ends per block"
    assert len(red) == BLOCKS and len(blue) == BLOCKS, "one adjacency per block end pair"
    degree = {}
    for node in NODES:
        degree[node] = 0
    for a, b in red:
        degree[a] = degree[a] + 1
        degree[b] = degree[b] + 1
    for node in NODES:
        assert degree[node] == 1, "node %s has %d red edges" % (node, degree[node])
    for node in NODES:
        degree[node] = 0
    for a, b in blue:
        degree[a] = degree[a] + 1
        degree[b] = degree[b] + 1
    for node in NODES:
        assert degree[node] == 1, "node %s has %d blue edges" % (node, degree[node])
    lines.append("every one of the %d block ends carries exactly one red and one blue edge"
                 % len(NODES))

    lines.append("P and Q share %d cycles to begin with"
                 % START_CYCLES)

    for step in STEPS:
        assert step["after"] == step["before"] + 1, (
            "a 2-break changed the cycle count from %d to %d"
            % (step["before"], step["after"]))
        assert len(step["removed"]) == 2 and len(step["added"]) == 2, (
            "a 2-break must swap exactly two edges for two")
        for edge in step["removed"]:
            assert edge in step["red_before"], "removed an edge P did not have"
        for edge in step["added"]:
            assert edge in step["red_after"], "added edge missing afterwards"
        ends_before = []
        for a, b in step["removed"]:
            ends_before.append(a)
            ends_before.append(b)
        ends_after = []
        for a, b in step["added"]:
            ends_after.append(a)
            ends_after.append(b)
        assert sorted(ends_before) == sorted(ends_after), (
            "a 2-break must rejoin the same four ends")
    lines.append("each of the %d 2-breaks rejoins the same four ends and adds one cycle"
                 % len(STEPS))

    assert len(STEPS) == BLOCKS - START_CYCLES, (
        "took %d 2-breaks but blocks minus cycles is %d"
        % (len(STEPS), BLOCKS - START_CYCLES))
    lines.append("the 2-break distance is blocks minus cycles = %d - %d = %d"
                 % (BLOCKS, START_CYCLES, len(STEPS)))

    final = STEPS[len(STEPS) - 1]["red_after"]
    assert final == blue, "P did not finish identical to Q"
    assert count_cycles(final, blue) == BLOCKS, (
        "the finished graph should have one cycle per block")
    lines.append("P finishes identical to Q, with all %d cycles trivial" % BLOCKS)

    assert CENTRE_Y - RADIUS - NODE_RADIUS > COUNT_Y + 0.35, (
        "circle collides with the counter")
    assert CENTRE_Y + RADIUS + NODE_RADIUS < FIGURE_HEIGHT - 0.15, (
        "circle runs off the top")
    # The block numbers ride outside the circle, so they are what the legend and
    # the counter actually have to clear, not the nodes.
    highest = 0.0
    lowest = FIGURE_HEIGHT
    block = 0
    while block < BLOCKS:
        tail = node_xy(NODES[2 * block])
        head = node_xy(NODES[2 * block + 1])
        mid_x = (tail[0] + head[0]) / 2
        mid_y = (tail[1] + head[1]) / 2
        outward = math.hypot(mid_x - CENTRE_X, mid_y - CENTRE_Y)
        label_y = CENTRE_Y + (mid_y - CENTRE_Y) / outward * (outward + BLOCK_LABEL_OUT)
        if label_y > highest:
            highest = label_y
        if label_y < lowest:
            lowest = label_y
        block = block + 1
    assert highest + 0.22 < LEGEND_Y, (
        "the top block number at %.2f in runs into the legend at %.2f in"
        % (highest, LEGEND_Y))
    assert lowest - 0.22 > COUNT_Y + 0.16, (
        "the bottom block number at %.2f in runs into the cycle count" % lowest)
    lines.append("circle of radius %.2f in sits clear of the counter" % RADIUS)
    return lines


def node_angle(index: int) -> float:
    return math.radians(90.0 - index * (360.0 / len(NODES)))


def node_xy(name: str) -> "tuple[float, float]":
    index = NODES.index(name)
    angle = node_angle(index)
    return (CENTRE_X + RADIUS * math.cos(angle), CENTRE_Y + RADIUS * math.sin(angle))


def trimmed(a: "tuple[float, float]", b: "tuple[float, float]") -> "tuple":
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length < 1e-9:
        return (a, b)
    unit_x = (b[0] - a[0]) / length
    unit_y = (b[1] - a[1]) / length
    trim = NODE_RADIUS + 0.03
    return ((a[0] + unit_x * trim, a[1] + unit_y * trim),
            (b[0] - unit_x * trim, b[1] - unit_y * trim))


def base_frame(duration: int) -> dict:
    return {"red": set(), "show_blue": True, "fading": [], "rising": [],
            "progress": 0.0, "cycles": 0, "distance": False, "cycle_of": {},
            "marked": [], "caption": "", "legend": False,
            "duration_ms": duration}


def build_specs() -> "list[dict]":
    specs = []
    blue = edge_set(BLUE_EDGES)

    only_q = base_frame(FIRST_HOLD_MS)
    only_q["red"] = set()
    only_q["cycles"] = 0
    only_q["caption"] = ("The target genome Q lays out the blocks, and its "
                         "adjacencies are the blue edges")
    only_q["legend"] = True
    specs.append(only_q)

    start_cycles = cycle_membership(edge_set(RED_EDGES), blue)
    arrive = base_frame(STEP_HOLD_MS)
    arrive["red"] = edge_set(RED_EDGES)
    arrive["cycles"] = START_CYCLES
    arrive["caption"] = ("Genome P's adjacencies go on the same nodes in red: red "
                         "is what has to change")
    arrive["legend"] = True
    specs.append(arrive)

    both = base_frame(STEP_HOLD_MS)
    both["red"] = edge_set(RED_EDGES)
    both["cycles"] = START_CYCLES
    both["cycle_of"] = dict(start_cycles)
    both["caption"] = ("Red and blue alternate, so the graph falls into %d cycles, "
                       "shown by the node colours" % START_CYCLES)
    both["legend"] = True
    specs.append(both)

    index = 0
    while index < len(STEPS):
        step = STEPS[index]

        mark = base_frame(MARK_MS)
        mark["red"] = set(step["red_before"])
        mark["cycles"] = step["before"]
        mark["cycle_of"] = dict(step["cycles_before"])
        mark["marked"] = list(step["removed"])
        mark["caption"] = "A 2-break cuts these two red edges"
        mark["legend"] = True
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
            moving["caption"] = "and rejoins the four loose ends the other way"
            moving["legend"] = True
            specs.append(moving)
            frame_index = frame_index + 1

        settled = base_frame(STEP_HOLD_MS)
        settled["red"] = set(step["red_after"])
        settled["cycles"] = step["after"]
        settled["cycle_of"] = dict(step["cycles_after"])
        settled["caption"] = ("One cycle has split in two, so the count goes from "
                              "%d to %d" % (step["before"], step["after"]))
        settled["legend"] = True
        specs.append(settled)
        index = index + 1

    final = base_frame(FINAL_HOLD_MS)
    final["red"] = set(STEPS[len(STEPS) - 1]["red_after"])
    final["cycles"] = BLOCKS
    final["cycle_of"] = cycle_membership(set(STEPS[len(STEPS) - 1]["red_after"]), blue)
    final["distance"] = True
    final["legend"] = True
    final["caption"] = ("Every cycle is now one red edge beside its blue one: P "
                        "became Q in %d 2-breaks, which is blocks minus the %d "
                        "cycles it started with" % (len(STEPS), START_CYCLES))
    specs.append(final)
    return specs


def draw_edge(axis: "plt.Axes", edge: "tuple[str, str]", colour: str,
              alpha: float, width: float, offset: float) -> None:
    """One edge, shifted a little to its own side of the line.

    Red and blue land on the same pair of nodes whenever P already agrees with Q,
    and at the end of the sorting that is every edge. Without the offset the red
    hides the blue and the finished graph looks like P alone.
    """
    if alpha <= 0.02:
        return
    a, b = trimmed(node_xy(edge[0]), node_xy(edge[1]))
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    shift_x = -(b[1] - a[1]) / length * offset
    shift_y = (b[0] - a[0]) / length * offset
    axis.plot([a[0] + shift_x, b[0] + shift_x], [a[1] + shift_y, b[1] + shift_y],
              color=colour, linewidth=width, solid_capstyle="round", zorder=3,
              alpha=alpha)


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)

    block = 0
    while block < BLOCKS:
        tail = NODES[2 * block]
        head = NODES[2 * block + 1]
        a, b = trimmed(node_xy(tail), node_xy(head))
        axis.plot([a[0], b[0]], [a[1], b[1]], color=DIM, linewidth=5.0,
                  solid_capstyle="round", zorder=2)
        mid_x = (node_xy(tail)[0] + node_xy(head)[0]) / 2
        mid_y = (node_xy(tail)[1] + node_xy(head)[1]) / 2
        outward = math.hypot(mid_x - CENTRE_X, mid_y - CENTRE_Y)
        axis.text(CENTRE_X + (mid_x - CENTRE_X) / outward * (outward + BLOCK_LABEL_OUT),
                  CENTRE_Y + (mid_y - CENTRE_Y) / outward * (outward + BLOCK_LABEL_OUT),
                  str(block + 1), fontsize=BLOCK_SIZE, color=INK, ha="center",
                  va="center", fontproperties=OPTIMA_ITALIC, zorder=6)
        block = block + 1

    if spec["show_blue"]:
        for edge in edge_set(BLUE_EDGES):
            draw_edge(axis, edge, BLUE, 1.0, EDGE_WIDTH, -EDGE_OFFSET)

    for edge in spec["red"]:
        draw_edge(axis, edge, RED, 1.0, EDGE_WIDTH, EDGE_OFFSET)
    for edge in spec["fading"]:
        draw_edge(axis, edge, RED, 1.0 - spec["progress"], EDGE_WIDTH,
                  EDGE_OFFSET)
    for edge in spec["rising"]:
        draw_edge(axis, edge, RED, spec["progress"], EDGE_WIDTH, EDGE_OFFSET)
    for edge in spec["marked"]:
        # A wider, darker line under the two edges about to be cut.
        draw_edge(axis, edge, "#7A0F13", 1.0, EDGE_WIDTH + 3.4, EDGE_OFFSET)
        draw_edge(axis, edge, RED, 1.0, EDGE_WIDTH, EDGE_OFFSET)

    for name in NODES:
        x, y = node_xy(name)
        if name in spec["cycle_of"]:
            outline = CYCLE_COLOURS[spec["cycle_of"][name] % len(CYCLE_COLOURS)]
            thickness = 2.6
        else:
            outline = INK
            thickness = 1.1
        circle = patches.Circle((x, y), NODE_RADIUS, facecolor="#FFFFFF",
                                edgecolor=outline, linewidth=thickness, zorder=5)
        axis.add_patch(circle)
        axis.text(x, y, name, fontsize=NODE_SIZE, color=INK, ha="center",
                  va="center", fontproperties=MONO, zorder=6)

    if spec["cycles"] > 0:
        axis.text(CENTRE_X, COUNT_Y, "%d cycles" % spec["cycles"],
                  fontsize=COUNT_SIZE, color=INK, ha="center", va="center",
                  fontproperties=MONO_BOLD, zorder=7)
    if spec["legend"]:
        axis.text(CENTRE_X - 1.15, LEGEND_Y, "red = P, the genome being sorted",
                  fontsize=LEGEND_SIZE, color=RED, ha="center", va="center",
                  fontproperties=OPTIMA_ITALIC, zorder=7)
        axis.text(CENTRE_X + 1.35, LEGEND_Y, "blue = Q, the target",
                  fontsize=LEGEND_SIZE, color=BLUE, ha="center", va="center",
                  fontproperties=OPTIMA_ITALIC, zorder=7)

    if spec["caption"] != "":
        axis.text(CENTRE_X, CAPTION_Y, spec["caption"], fontsize=LABEL_SIZE,
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
    directory = tempfile.mkdtemp(prefix="breakpoint_")
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
