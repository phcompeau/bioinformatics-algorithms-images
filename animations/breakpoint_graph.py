"""Animated 2-break sorting on the breakpoint graph, in the lecture style.

Follows the Fragile Genomes deck: build the breakpoint graph by drawing genome P's
adjacencies in red and genome Q's in blue on the same block ends. Every node has
one red and one blue edge, so the graph splits into alternating cycles. Each
2-break that splits a cycle brings P one step closer to Q, and the number of
2-breaks needed is blocks minus cycles.

Everything is asserted: the graph is a perfect alternating 2-regular graph, each
2-break raises the cycle count by exactly one, the number of 2-breaks equals
blocks minus the starting cycle count, and P's edges finish identical to Q's.

Run:  python3 breakpoint_graph.py OUTPUT.gif
"""

import math
import os
import sys
import tempfile

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, BLUE, DIM, FAINT, INK, MONO, MONO_BOLD,
                           OPTIMA_ITALIC, RED, ease, new_axes)
from make_gif import assemble_gif

BLOCKS = 4
# Node order around the circle: block tails and heads as Q lays them out.
NODES = ["1t", "1h", "2t", "2h", "3t", "3h", "4t", "4h"]
# P = (+1 -2 -3 +4), Q = (+1 +2 +3 +4), both circular.
RED_EDGES = [("1h", "2h"), ("2t", "3h"), ("3t", "4t"), ("4h", "1t")]
BLUE_EDGES = [("1h", "2t"), ("2h", "3t"), ("3h", "4t"), ("4h", "1t")]

FIGURE_WIDTH = 7.6
FIGURE_HEIGHT = 6.4
RENDER_DPI = 150
OUTPUT_WIDTH = 1140
OUTPUT_HEIGHT = 960

CENTRE_X = 3.8
CENTRE_Y = 3.55
RADIUS = 2.15
NODE_RADIUS = 0.155
NODE_SIZE = 10.0
BLOCK_SIZE = 14.0
EDGE_WIDTH = 2.2
COUNT_Y = 0.52
COUNT_SIZE = 26.0

STEP_HOLD_MS = 2600
BREAK_FRAMES = 10
BREAK_MS = 120
FIRST_HOLD_MS = 2600
FINAL_HOLD_MS = 5200


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
                      "red_before": set(red), "red_after": set(new_red)})
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
            "progress": 0.0, "cycles": 0, "distance": False,
            "duration_ms": duration}


def build_specs() -> "list[dict]":
    specs = []
    blue = edge_set(BLUE_EDGES)

    only_q = base_frame(FIRST_HOLD_MS)
    only_q["red"] = set()
    only_q["cycles"] = 0
    specs.append(only_q)

    both = base_frame(STEP_HOLD_MS)
    both["red"] = edge_set(RED_EDGES)
    both["cycles"] = START_CYCLES
    specs.append(both)

    for step in STEPS:
        frame_index = 1
        while frame_index <= BREAK_FRAMES:
            moving = base_frame(BREAK_MS)
            moving["red"] = set(step["red_before"]) - set(step["removed"])
            moving["fading"] = list(step["removed"])
            moving["rising"] = list(step["added"])
            moving["progress"] = ease(frame_index / BREAK_FRAMES)
            moving["cycles"] = step["before"]
            specs.append(moving)
            frame_index = frame_index + 1

        settled = base_frame(STEP_HOLD_MS)
        settled["red"] = set(step["red_after"])
        settled["cycles"] = step["after"]
        specs.append(settled)

    final = base_frame(FINAL_HOLD_MS)
    final["red"] = set(STEPS[len(STEPS) - 1]["red_after"])
    final["cycles"] = BLOCKS
    final["distance"] = True
    specs.append(final)
    return specs


def draw_edge(axis: "plt.Axes", edge: "tuple[str, str]", colour: str,
              alpha: float, width: float) -> None:
    if alpha <= 0.02:
        return
    a, b = trimmed(node_xy(edge[0]), node_xy(edge[1]))
    axis.plot([a[0], b[0]], [a[1], b[1]], color=colour, linewidth=width,
              solid_capstyle="round", zorder=3, alpha=alpha)


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
        axis.text(CENTRE_X + (mid_x - CENTRE_X) / outward * (outward + 0.42),
                  CENTRE_Y + (mid_y - CENTRE_Y) / outward * (outward + 0.42),
                  str(block + 1), fontsize=BLOCK_SIZE, color=INK, ha="center",
                  va="center", fontproperties=OPTIMA_ITALIC, zorder=6)
        block = block + 1

    if spec["show_blue"]:
        for edge in edge_set(BLUE_EDGES):
            draw_edge(axis, edge, BLUE, 1.0, EDGE_WIDTH)

    for edge in spec["red"]:
        draw_edge(axis, edge, RED, 1.0, EDGE_WIDTH)
    for edge in spec["fading"]:
        draw_edge(axis, edge, RED, 1.0 - spec["progress"], EDGE_WIDTH)
    for edge in spec["rising"]:
        draw_edge(axis, edge, RED, spec["progress"], EDGE_WIDTH)

    for name in NODES:
        x, y = node_xy(name)
        circle = patches.Circle((x, y), NODE_RADIUS, facecolor="#FFFFFF",
                                edgecolor=INK, linewidth=1.1, zorder=5)
        axis.add_patch(circle)
        axis.text(x, y, name, fontsize=NODE_SIZE, color=INK, ha="center",
                  va="center", fontproperties=MONO, zorder=6)

    if spec["cycles"] > 0:
        axis.text(CENTRE_X, COUNT_Y, str(spec["cycles"]), fontsize=COUNT_SIZE,
                  color=INK, ha="center", va="center", fontproperties=MONO_BOLD,
                  zorder=7)
    if spec["distance"]:
        axis.text(CENTRE_X, COUNT_Y + 0.52, "%d - %d = %d"
                  % (BLOCKS, START_CYCLES, len(STEPS)), fontsize=COUNT_SIZE - 8,
                  color=DIM, ha="center", va="center", fontproperties=MONO,
                  zorder=7)

    figure.savefig(output_path, facecolor=BACKGROUND)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: breakpoint_graph.py OUTPUT.gif")
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

    assemble_gif(frame_paths, sys.argv[1], width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT,
                 frame_durations=durations)
    print("Saved " + sys.argv[1])


if __name__ == "__main__":
    main()
