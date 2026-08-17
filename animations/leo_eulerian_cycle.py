"""Animated Eulerian cycle construction ("Leo the ant") in the Rosalind style.

Reproduces the Chapter 3 ant sequence (Master Figures.pptx slides 49-54) as a
looping GIF. The graph geometry and edge list are lifted from the source deck,
not hand-typed by eye, and the three-act walk is asserted for correctness
before any frame is drawn.

Act 1 (green): Leo starts at a random node, walks until he gets stuck, and
               discovers he can only get stuck where he started.
Act 2 (blue):  He restarts from a node on the cycle that still has an unused
               edge, re-walks the cycle, then keeps going. Stuck again.
Act 3 (red):   Same move once more, and this time every edge gets used.

Run:  python3 leo_eulerian_cycle.py OUTPUT.gif
"""

import math
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image

from make_gif import assemble_gif

BLUE = "#176FC1"
GREEN = "#149B52"
RED = "#ED1C24"
EDGE_GRAY = "#404040"
NODE_GRAY = "#D4D4D4"
SHADOW = "#B7B7B7"
TEXT_DARK = "#262626"

OPTIMA = font_manager.FontProperties(family="Optima")
OPTIMA_ITALIC = font_manager.FontProperties(family="Optima", style="italic")

# Node centers measured from Master Figures.pptx slide 49 (inches from the
# top-left of a 10 x 7.5 slide), then recentered and scaled up for the GIF.
DECK_NODES = {
    "A": (6.53, 3.59),
    "B": (6.37, 4.62),
    "C": (8.04, 4.40),
    "D": (5.28, 4.81),
    "E": (8.04, 5.90),
    "F": (9.16, 5.28),
    "G": (6.17, 6.35),
}

# Edge list recovered from the deck's connectors (endpoint proximity plus
# flipH/flipV plus which end carries the arrowhead). One curved freeform.
EDGES = [
    ("A", "C"),
    ("A", "B"),
    ("B", "C"),
    ("C", "E"),
    ("C", "A"),
    ("D", "A"),
    ("D", "G"),
    ("E", "D"),
    ("E", "F"),
    ("F", "D"),
    ("G", "E"),
]
CURVED_EDGE = ("C", "A")
CURVE_BOW = 0.30

GRAPH_SCALE = 1.62
GRAPH_CENTER = (5.0, 3.15)
NODE_RADIUS = 0.20 * GRAPH_SCALE
EDGE_GAP = 0.065 * GRAPH_SCALE
ANT_EXTENT = 0.86
ANT_NATIVE_HEADING = 44.0

CAPTION_Y = 6.92
COUNTER_Y = 6.40

# Number placement is solved rather than hand-tuned: the middle of this graph
# has four edges crossing, and eyeballed offsets kept colliding.
NUMBER_FRACTIONS = [0.5, 0.38, 0.62, 0.28, 0.72]
NUMBER_SIDES = [-1.0, 1.0]
NUMBER_OFFSETS = [0.30, 0.40, 0.50]
OWN_EDGE_MAX = 0.58
OTHER_EDGE_CLEARANCE = 0.19
LABEL_CLEARANCE = 0.44
NODE_CLEARANCE = 0.15

ACT_COLORS = {1: GREEN, 2: BLUE, 3: RED}

ACT_ONE_START = "G"
ACT_TWO_START = "E"
ACT_THREE_START = "C"

ACT_ONE_WALK = [("G", "E"), ("E", "F"), ("F", "D"), ("D", "G")]
ACT_TWO_WALK = [
    ("E", "F"), ("F", "D"), ("D", "G"), ("G", "E"),
    ("E", "D"), ("D", "A"), ("A", "C"), ("C", "E"),
]
ACT_THREE_WALK = [
    ("C", "E"), ("E", "F"), ("F", "D"), ("D", "G"),
    ("G", "E"), ("E", "D"), ("D", "A"), ("A", "C"),
    ("C", "A"), ("A", "B"), ("B", "C"),
]

FRAMES_NEW_EDGE = 11
FRAMES_REWALK = 6

# Pacing. A caption's time on screen is driven by how long it takes to read,
# never by how many frames its animation happens to need: a sentence that
# appears for one frame is invisible no matter how pretty the frame is.
# GIF delays are stored in centiseconds, so keep every duration a multiple
# of 10 or the real playback drifts shorter than the number reported here.
MOTION_MS = 110
READING_BASE_MS = 500
READING_PER_CHAR_MS = 42
READING_MIN_MS = 1500
READING_MAX_MS = 3200
REPEAT_CAPTION_FACTOR = 0.6
RENDER_DPI = 120
OUTPUT_WIDTH = 1200
OUTPUT_HEIGHT = 900

NODE_ARRIVAL_MS = 300
REWALK_ARRIVAL_MS = 130
FINAL_HOLD_MS = 4200


def node_positions() -> "dict[str, tuple[float, float]]":
    """Deck coordinates flipped to y-up, recentered, and scaled."""
    xs = []
    ys = []
    for name in DECK_NODES:
        xs.append(DECK_NODES[name][0])
        ys.append(7.5 - DECK_NODES[name][1])
    deck_center_x = (min(xs) + max(xs)) / 2
    deck_center_y = (min(ys) + max(ys)) / 2
    positions = {}
    for name in DECK_NODES:
        deck_x = DECK_NODES[name][0]
        deck_y = 7.5 - DECK_NODES[name][1]
        x = GRAPH_CENTER[0] + (deck_x - deck_center_x) * GRAPH_SCALE
        y = GRAPH_CENTER[1] + (deck_y - deck_center_y) * GRAPH_SCALE
        positions[name] = (x, y)
    return positions


NODES = node_positions()


def edge_polyline(edge: "tuple[str, str]") -> "np.ndarray":
    """Dense polyline for an edge, from node center to node center."""
    a = np.array(NODES[edge[0]], dtype=float)
    b = np.array(NODES[edge[1]], dtype=float)
    ts = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
    if edge != CURVED_EDGE:
        return a + (b - a) * ts
    delta = b - a
    length = float(np.hypot(delta[0], delta[1]))
    perp = np.array([delta[1], -delta[0]]) / length
    control = (a + b) / 2 + perp * (CURVE_BOW * length)
    return (1 - ts) ** 2 * a + 2 * (1 - ts) * ts * control + ts ** 2 * b


def trim_polyline(points: "np.ndarray") -> "np.ndarray":
    """Drop the ends that fall inside a node plus its edge gap."""
    clearance = NODE_RADIUS + EDGE_GAP
    start = points[0]
    end = points[-1]
    from_start = np.hypot(points[:, 0] - start[0], points[:, 1] - start[1])
    from_end = np.hypot(points[:, 0] - end[0], points[:, 1] - end[1])
    keep = np.logical_and(from_start >= clearance, from_end >= clearance)
    indices = np.nonzero(keep)[0]
    if len(indices) < 2:
        return points[[0, -1]]
    return points[indices[0]:indices[-1] + 1]


def build_edge_paths() -> "dict[tuple[str, str], np.ndarray]":
    """Trimmed drawing/travel polyline for every edge."""
    paths = {}
    for edge in EDGES:
        paths[edge] = trim_polyline(edge_polyline(edge))
    return paths


EDGE_PATHS = build_edge_paths()


def point_along(edge: "tuple[str, str]", fraction: float) -> "tuple[float, float, float]":
    """Position and heading (degrees, y-up) at a fraction of an edge's length."""
    points = EDGE_PATHS[edge]
    deltas = np.diff(points, axis=0)
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    target = cumulative[-1] * min(max(fraction, 0.0), 1.0)
    index = int(np.searchsorted(cumulative, target))
    if index >= len(points) - 1:
        index = len(points) - 2
    span = cumulative[index + 1] - cumulative[index]
    if span <= 0:
        local = 0.0
    else:
        local = (target - cumulative[index]) / span
    x = points[index][0] + (points[index + 1][0] - points[index][0]) * local
    y = points[index][1] + (points[index + 1][1] - points[index][1]) * local
    heading = math.degrees(math.atan2(deltas[index][1], deltas[index][0]))
    return (x, y, heading)


def distance_to_edge(point: "tuple[float, float]", edge: "tuple[str, str]") -> float:
    """Shortest distance from a point to an edge's drawn polyline."""
    points = EDGE_PATHS[edge]
    return float(np.min(np.hypot(points[:, 0] - point[0], points[:, 1] - point[1])))


def candidate_score(point: "tuple[float, float]", edge: "tuple[str, str]",
                    placed: dict) -> float:
    """How much room a candidate number position has. Higher is better."""
    own = distance_to_edge(point, edge)
    if own > OWN_EDGE_MAX:
        return -1.0
    worst = own
    for other in EDGES:
        if other == edge:
            continue
        gap = distance_to_edge(point, other)
        if gap < worst:
            worst = gap
    for name in NODES:
        gap = math.hypot(point[0] - NODES[name][0], point[1] - NODES[name][1])
        gap = gap - NODE_RADIUS
        if gap < worst:
            worst = gap
    for other_edge in placed:
        spot = placed[other_edge]
        gap = math.hypot(point[0] - spot[0], point[1] - spot[1]) / 2.4
        if gap < worst:
            worst = gap
    return worst


def place_numbers() -> "dict[tuple[str, str], tuple[float, float]]":
    """Pick a position for every edge number, greedily maximizing clearance."""
    placed = {}
    for edge in EDGES:
        best_point = (0.0, 0.0)
        best_score = -2.0
        for fraction in NUMBER_FRACTIONS:
            anchor = point_along(edge, fraction)
            heading = math.radians(anchor[2])
            for side in NUMBER_SIDES:
                perp_x = -math.sin(heading) * side
                perp_y = math.cos(heading) * side
                for offset in NUMBER_OFFSETS:
                    point = (anchor[0] + perp_x * offset, anchor[1] + perp_y * offset)
                    score = candidate_score(point, edge, placed)
                    if score > best_score:
                        best_score = score
                        best_point = point
        placed[edge] = best_point
    return placed


NUMBER_POSITIONS = place_numbers()


def verify_numbers() -> "list[str]":
    """Assert no number sits on a wrong edge, a node, or another number."""
    lines = []
    for edge in EDGES:
        point = NUMBER_POSITIONS[edge]
        own = distance_to_edge(point, edge)
        assert own <= OWN_EDGE_MAX, (
            "number for %s drifted %.2f from its own edge" % (edge, own))
        for other in EDGES:
            if other == edge:
                continue
            gap = distance_to_edge(point, other)
            assert gap >= OTHER_EDGE_CLEARANCE, (
                "number for %s sits %.2f from edge %s" % (edge, gap, other))
        for name in NODES:
            gap = math.hypot(point[0] - NODES[name][0], point[1] - NODES[name][1])
            assert gap >= NODE_RADIUS + NODE_CLEARANCE, (
                "number for %s sits %.2f from node %s" % (edge, gap, name))
        assert 0.4 <= point[0] <= 9.6, "number for %s is off canvas" % (edge,)
        assert 0.4 <= point[1] <= 6.0, "number for %s is off canvas" % (edge,)
    index = 0
    while index < len(EDGES):
        other_index = index + 1
        while other_index < len(EDGES):
            a = NUMBER_POSITIONS[EDGES[index]]
            b = NUMBER_POSITIONS[EDGES[other_index]]
            gap = math.hypot(a[0] - b[0], a[1] - b[1])
            assert gap >= LABEL_CLEARANCE, (
                "numbers for %s and %s are %.2f apart"
                % (EDGES[index], EDGES[other_index], gap))
            other_index = other_index + 1
        index = index + 1
    lines.append("all 11 numbers clear of every other edge, node, and number")
    return lines


def verify_graph() -> "list[str]":
    """Assert the graph and all three walks are correct. Returns check lines."""
    lines = []

    assert len(EDGES) == len(set(EDGES)), "duplicate edge in the edge list"
    assert len(EDGES) == 11, "expected 11 edges, got %d" % len(EDGES)
    assert len(NODES) == 7, "expected 7 nodes, got %d" % len(NODES)
    lines.append("7 nodes, 11 distinct edges")

    out_degree = {}
    in_degree = {}
    for name in NODES:
        out_degree[name] = 0
        in_degree[name] = 0
    for edge in EDGES:
        assert edge[0] in NODES and edge[1] in NODES, "edge touches unknown node"
        out_degree[edge[0]] = out_degree[edge[0]] + 1
        in_degree[edge[1]] = in_degree[edge[1]] + 1
    for name in NODES:
        assert out_degree[name] == in_degree[name], (
            "node %s unbalanced: in %d out %d" % (name, in_degree[name], out_degree[name]))
    lines.append("every node balanced (in-degree == out-degree), so a cycle exists")

    reachable = set()
    frontier = ["A"]
    while len(frontier) > 0:
        current = frontier.pop()
        if current in reachable:
            continue
        reachable.add(current)
        for edge in EDGES:
            if edge[0] == current:
                frontier.append(edge[1])
    assert reachable == set(NODES), "graph is not strongly connected enough: %s" % reachable
    lines.append("all 7 nodes reachable, graph is connected")

    acts = [
        (1, ACT_ONE_START, ACT_ONE_WALK),
        (2, ACT_TWO_START, ACT_TWO_WALK),
        (3, ACT_THREE_START, ACT_THREE_WALK),
    ]
    edge_set = set(EDGES)
    for act_number, start, walk in acts:
        assert len(walk) == len(set(walk)), "act %d repeats an edge" % act_number
        assert walk[0][0] == start, "act %d does not start at %s" % (act_number, start)
        assert walk[-1][1] == start, "act %d does not return to %s" % (act_number, start)
        step = 0
        while step < len(walk):
            assert walk[step] in edge_set, (
                "act %d step %d is not a real edge: %s" % (act_number, step, walk[step]))
            if step > 0:
                assert walk[step - 1][1] == walk[step][0], (
                    "act %d breaks at step %d" % (act_number, step))
            step = step + 1
        lines.append("act %d: valid closed walk of %d edges from %s"
                     % (act_number, len(walk), start))

    for act_number, start, walk in acts:
        used = set(walk)
        stranded = 0
        for name in NODES:
            for edge in EDGES:
                if edge[0] == name and edge not in used:
                    stranded = stranded + 1
        if act_number < 3:
            assert stranded > 0, "act %d should leave unused edges" % act_number
    lines.append("acts 1 and 2 both leave unused edges, so restarting is forced")

    assert set(ACT_ONE_WALK).issubset(set(ACT_TWO_WALK)), "act 2 loses act 1's edges"
    assert set(ACT_TWO_WALK).issubset(set(ACT_THREE_WALK)), "act 3 loses act 2's edges"
    lines.append("each act keeps every edge the previous act built")

    # The node each act restarts from must have had an unused edge at that moment.
    after_one = set(ACT_ONE_WALK)
    open_at_two = False
    for edge in EDGES:
        if edge[0] == ACT_TWO_START and edge not in after_one:
            open_at_two = True
    assert open_at_two, "act 2 restarts at a node with no unused edge"
    after_two = set(ACT_TWO_WALK)
    open_at_three = False
    for edge in EDGES:
        if edge[0] == ACT_THREE_START and edge not in after_two:
            open_at_three = True
    assert open_at_three, "act 3 restarts at a node with no unused edge"
    lines.append("each restart node genuinely had an unused edge")

    assert set(ACT_THREE_WALK) == edge_set, "act 3 is not an Eulerian cycle"
    assert len(ACT_THREE_WALK) == 11, "act 3 does not have 11 steps"
    lines.append("act 3 uses all 11 edges exactly once and returns to C: Eulerian cycle")

    for edge in EDGES:
        points = EDGE_PATHS[edge]
        a = np.array(NODES[edge[0]])
        b = np.array(NODES[edge[1]])
        gap_start = float(np.hypot(points[0][0] - a[0], points[0][1] - a[1]))
        gap_end = float(np.hypot(points[-1][0] - b[0], points[-1][1] - b[1]))
        assert gap_start >= NODE_RADIUS, "edge %s starts inside node %s" % (edge, edge[0])
        assert gap_end >= NODE_RADIUS, "edge %s ends inside node %s" % (edge, edge[1])
        assert gap_start < NODE_RADIUS + 3 * EDGE_GAP, "edge %s starts too far out" % (edge,)
        assert gap_end < NODE_RADIUS + 3 * EDGE_GAP, "edge %s ends too far out" % (edge,)
    lines.append("every edge stops just short of both nodes, none overlapping")

    return lines


def load_ant() -> "Image.Image":
    """Ant glyph from the deck, white made transparent, padded to a square."""
    here = os.path.dirname(os.path.abspath(__file__))
    ant_path = os.path.join(here, "assets", "leo_the_ant.png")
    ant = Image.open(ant_path).convert("RGBA")
    pixels = np.array(ant)
    brightness = pixels[:, :, 0:3].mean(axis=2)
    # Background is either near-white or already transparent; the ant is the dark
    # ink that survives both tests.
    is_background = np.logical_or(brightness > 200, pixels[:, :, 3] < 20)
    pixels[:, :, 3] = np.where(is_background, 0, 255).astype(np.uint8)
    pixels[:, :, 0] = 38
    pixels[:, :, 1] = 38
    pixels[:, :, 2] = 38
    ant = Image.fromarray(pixels)
    columns = np.nonzero(pixels[:, :, 3].sum(axis=0))[0]
    rows = np.nonzero(pixels[:, :, 3].sum(axis=1))[0]
    ant = ant.crop((int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1))
    side = int(math.hypot(ant.width, ant.height)) + 2
    square = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    square.paste(ant, ((side - ant.width) // 2, (side - ant.height) // 2), ant)
    return square.resize((260, 260), Image.LANCZOS)


ANT_IMAGE = load_ant()


def draw_node(axis: "plt.Axes", name: str, ring_color: str) -> None:
    """Shadowed gray circle; a colored ring marks the current act's start node."""
    x = NODES[name][0]
    y = NODES[name][1]
    shadow = patches.Circle((x + 0.045, y - 0.045), NODE_RADIUS, facecolor=SHADOW,
                            edgecolor="none", zorder=4)
    axis.add_patch(shadow)
    if ring_color == "":
        circle = patches.Circle((x, y), NODE_RADIUS, facecolor=NODE_GRAY,
                                edgecolor=EDGE_GRAY, linewidth=1.6, zorder=5)
    else:
        circle = patches.Circle((x, y), NODE_RADIUS, facecolor=ring_color,
                                edgecolor=ring_color, linewidth=2.6, alpha=1.0, zorder=5)
        circle.set_facecolor(lighten(ring_color))
    axis.add_patch(circle)


def lighten(color: str) -> str:
    """Pale version of a palette color, for start-node fills."""
    if color == GREEN:
        return "#BFE6CE"
    if color == BLUE:
        return "#C3DCF0"
    if color == RED:
        return "#F9C4C6"
    return NODE_GRAY


def draw_edge(axis: "plt.Axes", edge: "tuple[str, str]", color: str, number: int) -> None:
    """One edge, plus its traversal number once it has been walked."""
    points = EDGE_PATHS[edge]
    if color == "":
        line_color = EDGE_GRAY
        width = 2.4
        zorder = 2
    else:
        line_color = color
        width = 3.4
        zorder = 3
    arrow = patches.FancyArrowPatch(
        path=MplPath(points), arrowstyle="-|>", mutation_scale=17,
        linewidth=width, color=line_color, zorder=zorder,
        joinstyle="round", capstyle="round")
    axis.add_patch(arrow)
    if number == 0:
        return
    spot = NUMBER_POSITIONS[edge]
    axis.text(spot[0], spot[1], str(number),
              fontsize=17, color=color, ha="center", va="center",
              fontproperties=OPTIMA, zorder=7)


def draw_ant(axis: "plt.Axes", x: float, y: float, heading: float) -> None:
    """Place the rotated ant glyph centered on (x, y), facing `heading`."""
    rotated = ANT_IMAGE.rotate(heading - ANT_NATIVE_HEADING, resample=Image.BICUBIC,
                               expand=False)
    half = ANT_EXTENT / 2
    axis.imshow(np.array(rotated), extent=(x - half, x + half, y - half, y + half),
                zorder=9, interpolation="bilinear")


def draw_frame(spec: dict, output_path: str) -> None:
    """Render one frame from a state dict produced by build_frame_specs."""
    figure, axis = plt.subplots(figsize=(10, 7.5), dpi=RENDER_DPI)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 7.5)
    axis.set_aspect("equal")
    axis.axis("off")

    axis.text(5.0, CAPTION_Y, spec["caption"], fontsize=17, color=TEXT_DARK,
              ha="center", va="center", fontproperties=OPTIMA_ITALIC)
    counter = "%d of 11 edges used" % spec["used_count"]
    axis.text(5.0, COUNTER_Y, counter, fontsize=14, color=spec["counter_color"],
              ha="center", va="center", fontproperties=OPTIMA)

    for edge in EDGES:
        color = spec["edge_colors"].get(edge, "")
        number = spec["edge_numbers"].get(edge, 0)
        draw_edge(axis, edge, color, number)

    for name in NODES:
        if name == spec["start_node"]:
            draw_node(axis, name, ACT_COLORS[spec["act"]])
        else:
            draw_node(axis, name, "")

    draw_ant(axis, spec["ant"][0], spec["ant"][1], spec["ant"][2])

    figure.savefig(output_path, facecolor="white")
    plt.close(figure)


def ease_step(fraction: float) -> float:
    """Gentle start and stop for a single edge crossing."""
    return 0.5 * (1 - math.cos(math.pi * min(max(fraction, 0.0), 1.0)))


def reading_duration(caption: str, already_seen: bool) -> int:
    """How long a caption must stay still to be readable, from its length."""
    milliseconds = READING_BASE_MS + READING_PER_CHAR_MS * len(caption)
    if milliseconds < READING_MIN_MS:
        milliseconds = READING_MIN_MS
    if milliseconds > READING_MAX_MS:
        milliseconds = READING_MAX_MS
    if already_seen:
        milliseconds = int(milliseconds * REPEAT_CAPTION_FACTOR)
    return int(round(milliseconds / 10.0) * 10)


def make_spec(act_number: int, caption: str, edge_colors: dict, edge_numbers: dict,
              start: str, used_count: int, ant: "tuple[float, float, float]",
              duration_ms: int) -> dict:
    """One frame's full state plus how long it stays on screen."""
    return {
        "act": act_number,
        "caption": caption,
        "counter_color": ACT_COLORS[act_number],
        "edge_colors": dict(edge_colors),
        "edge_numbers": dict(edge_numbers),
        "start_node": start,
        "used_count": used_count,
        "ant": ant,
        "duration_ms": duration_ms,
    }


def build_frame_specs() -> "list[dict]":
    """Simulate all three acts and emit one state dict per frame.

    Motion frames run at a steady rate. Everything else is a still frame whose
    duration is chosen for a human: long enough to read a caption that just
    changed, or to register that the ant has arrived somewhere and has a
    decision to make.
    """
    specs = []
    edge_colors = {}
    seen_captions = set()
    current_caption = ""
    acts = [
        (1, ACT_ONE_START, ACT_ONE_WALK),
        (2, ACT_TWO_START, ACT_TWO_WALK),
        (3, ACT_THREE_START, ACT_THREE_WALK),
    ]
    act_index = 0
    while act_index < len(acts):
        act_number = acts[act_index][0]
        start = acts[act_index][1]
        walk = acts[act_index][2]
        color = ACT_COLORS[act_number]
        edge_numbers = {}
        used = set()
        for edge in edge_colors:
            used.add(edge)
        ant_node = start
        heading = resting_heading(start)

        step = 0
        while step < len(walk):
            edge = walk[step]
            is_new = edge not in edge_colors

            caption = block_caption(act_number, is_new)
            if caption != current_caption:
                resting = (NODES[ant_node][0], NODES[ant_node][1], heading)
                specs.append(make_spec(
                    act_number, caption, edge_colors, edge_numbers, start,
                    len(used), resting,
                    reading_duration(caption, caption in seen_captions)))
                seen_captions.add(caption)
                current_caption = caption

            if is_new:
                frame_count = FRAMES_NEW_EDGE
            else:
                frame_count = FRAMES_REWALK
            edge_numbers[edge] = step + 1
            if is_new:
                edge_colors[edge] = color
                used.add(edge)

            frame = 0
            while frame < frame_count:
                fraction = ease_step((frame + 1) / frame_count)
                position = point_along(edge, fraction)
                specs.append(make_spec(
                    act_number, caption, edge_colors, edge_numbers, start,
                    len(used), position, MOTION_MS))
                heading = position[2]
                frame = frame + 1

            # Beat on arrival: the ant is standing on a node, deciding.
            ant_node = edge[1]
            if is_new:
                arrival_ms = NODE_ARRIVAL_MS
            else:
                arrival_ms = REWALK_ARRIVAL_MS
            specs.append(make_spec(
                act_number, caption, edge_colors, edge_numbers, start, len(used),
                (NODES[ant_node][0], NODES[ant_node][1], heading), arrival_ms))
            step = step + 1

        caption = closing_caption(act_number, len(used))
        if act_number == 3:
            closing_ms = FINAL_HOLD_MS
        else:
            closing_ms = reading_duration(caption, caption in seen_captions)
        specs.append(make_spec(
            act_number, caption, edge_colors, edge_numbers, start, len(used),
            (NODES[ant_node][0], NODES[ant_node][1], heading), closing_ms))
        seen_captions.add(caption)
        current_caption = caption
        act_index = act_index + 1
    return specs


def resting_heading(node: str) -> float:
    """Point a resting ant roughly into the graph so it never faces off-canvas."""
    toward_x = GRAPH_CENTER[0] - NODES[node][0]
    toward_y = GRAPH_CENTER[1] - NODES[node][1]
    return math.degrees(math.atan2(toward_y, toward_x))


def block_caption(act_number: int, is_new: bool) -> str:
    """One caption per block of traversals, so text never flips mid-motion."""
    if act_number == 1:
        return "Leo picks a node at random and walks along unused edges."
    if is_new:
        return "Back at the start, so continue onto edges the cycle never used."
    return "Restart where an unused edge remains, and re-walk the cycle from there."


def closing_caption(act_number: int, used_count: int) -> str:
    if act_number == 3:
        return "All 11 edges used, ending where he started: an Eulerian cycle."
    return "Stuck, and only at the node he started from. %d of 11 edges used." % used_count


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: leo_eulerian_cycle.py OUTPUT.gif")
        return

    print("Structural checks:")
    for line in verify_graph():
        print("  ok: " + line)
    for line in verify_numbers():
        print("  ok: " + line)

    specs = build_frame_specs()
    durations = []
    for spec in specs:
        durations.append(spec["duration_ms"])
    print("Rendering %d frames at %dx%d, %.1f s of playback..."
          % (len(specs), OUTPUT_WIDTH, OUTPUT_HEIGHT, sum(durations) / 1000.0))
    directory = tempfile.mkdtemp(prefix="leo_frames_")
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
    print("Frames kept in " + directory)


if __name__ == "__main__":
    main()
