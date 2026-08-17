"""Animated Manhattan network dynamic programming, in the Rosalind style.

Two animations from Chapter 5, sharing one lattice and one verified DP table:

  fill      - the network fills in node by node. Each interior node shows both
              incoming candidates (blue from the left, green from above), then
              commits to the larger and writes its score.
  backtrack - starting from the sink, follow each node's winning edge back to
              the source, building the longest path in red.

Edge weights are transcribed from the published Chapter 5 figures, and the DP
is recomputed here and asserted against the published node values and the
published red path, so a wrong number cannot survive to a frame.

Run:  python3 manhattan_dp.py fill OUTPUT.gif
      python3 manhattan_dp.py backtrack OUTPUT.gif
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

# Render the book's s_{i,j} subscripts in Optima rather than the mathtext default.
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Optima"
plt.rcParams["mathtext.it"] = "Optima:italic"
plt.rcParams["mathtext.bf"] = "Optima:bold"

from make_gif import assemble_gif

BLUE = "#176FC1"
GREEN = "#149B52"
RED = "#ED1C24"
EDGE_GRAY = "#595959"
FADED_GRAY = "#BFBFBF"
NODE_FILL = "#D9D9D9"
NODE_EDGE = "#404040"
SHADOW = "#B7B7B7"
TEXT_DARK = "#262626"

OPTIMA = font_manager.FontProperties(family="Optima")
OPTIMA_BOLD = font_manager.FontProperties(family="Optima", weight="bold")
OPTIMA_ITALIC = font_manager.FontProperties(family="Optima", style="italic")

ROWS = 5
COLS = 5

# Weight of the rightward edge out of (row, col); 5 rows of 4.
RIGHT_WEIGHTS = [
    [3, 2, 4, 0],
    [3, 2, 4, 2],
    [0, 7, 3, 4],
    [3, 3, 0, 2],
    [1, 3, 2, 2],
]

# Weight of the downward edge out of (row, col); 4 rows of 5.
DOWN_WEIGHTS = [
    [1, 0, 2, 4, 3],
    [4, 6, 5, 2, 1],
    [4, 4, 5, 2, 1],
    [5, 6, 8, 5, 3],
]

# Node scores as printed in manhattan_revisited_grid-5.png and backtracking.png.
PUBLISHED_VALUES = [
    [0, 3, 5, 9, 9],
    [1, 4, 7, 13, 15],
    [5, 10, 17, 20, 24],
    [9, 14, 22, 22, 25],
    [14, 20, 30, 32, 34],
]

# The red path in backtracking.png, source to sink.
PUBLISHED_PATH = [
    (0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4),
]

RENDER_DPI = 120
OUTPUT_WIDTH = 1200
OUTPUT_HEIGHT = 900

SPACING = 1.12
NODE_RADIUS = 0.26
# The fill needs room under the grid for the recurrence block; the backtrack has
# no block, so its grid is recentered in the taller space.
GRID_TOP_FILL = 6.45
GRID_TOP_BACKTRACK = 5.80
GRID_TOP = GRID_TOP_FILL
GRID_LEFT = (10.0 - SPACING * (COLS - 1)) / 2
EDGE_GAP = 0.07
WEIGHT_OFFSET = 0.21

CAPTION_Y = 7.22
COUNTER_Y = 6.88

# The recurrence sits under the grid, laid out like the book's figure:
# "s_{i,j} = max {" then one line per incoming edge, then aligned = columns.
PREFIX_RIGHT_X = 1.48
BRACE_X = 1.62
TERM_X = 1.82
COL1_X = 6.80
COL2_X = 7.80
BLOCK_MIDDLE_Y = 0.80
BLOCK_LINE_Y = [1.05, 0.55]
PREFIX_SIZE = 13.0
TERM_SIZE = 11.0
COLUMN_SIZE = 12.0

MOTION_MS = 110
READING_BASE_MS = 500
READING_PER_CHAR_MS = 42
READING_MIN_MS = 1200
READING_MAX_MS = 3200
REPEAT_CAPTION_FACTOR = 0.55
INIT_MS = 430
FINAL_HOLD_MS = 4600

# The panel shows fresh arithmetic at every step, so these frames need reading
# time exactly like a caption does. The dwell decays: the first few nodes carry
# the explanation, and once the pattern is familiar the same work reads faster.
CONSIDER_SCHEDULE = [(3, 2200), (7, 1600)]
CONSIDER_MS_LATE = 1100
COMMIT_SCHEDULE = [(7, 700)]
COMMIT_MS_LATE = 500
BACKTRACK_SCHEDULE = [(2, 2000)]
BACKTRACK_MS_LATE = 1400


def scheduled_duration(step_index: int, schedule: "list[tuple[int, int]]",
                       late_ms: int) -> int:
    """Dwell for the nth step of a repeating pattern, decaying as it repeats."""
    for limit, milliseconds in schedule:
        if step_index < limit:
            return milliseconds
    return late_ms


def node_xy(row: int, col: int) -> "tuple[float, float]":
    """Center of the lattice node at (row, col); row 0 is the top."""
    return (GRID_LEFT + col * SPACING, GRID_TOP - row * SPACING)


def compute_dp() -> "tuple[list, dict]":
    """Longest-path scores and each node's winning incoming edge.

    Ties go to the horizontal edge, which is how the published figures break
    the tie at (3, 3), where both predecessors offer 22.
    """
    values = []
    row = 0
    while row < ROWS:
        values.append([0] * COLS)
        row = row + 1
    choice = {}
    row = 0
    while row < ROWS:
        col = 0
        while col < COLS:
            if row == 0 and col == 0:
                values[row][col] = 0
            elif row == 0:
                values[row][col] = values[row][col - 1] + RIGHT_WEIGHTS[row][col - 1]
                choice[(row, col)] = "right"
            elif col == 0:
                values[row][col] = values[row - 1][col] + DOWN_WEIGHTS[row - 1][col]
                choice[(row, col)] = "down"
            else:
                from_left = values[row][col - 1] + RIGHT_WEIGHTS[row][col - 1]
                from_above = values[row - 1][col] + DOWN_WEIGHTS[row - 1][col]
                if from_left >= from_above:
                    values[row][col] = from_left
                    choice[(row, col)] = "right"
                else:
                    values[row][col] = from_above
                    choice[(row, col)] = "down"
            col = col + 1
        row = row + 1
    return (values, choice)


VALUES, CHOICE = compute_dp()


def backtrack_path() -> "list[tuple[int, int]]":
    """Longest path, source to sink, by following winning edges backwards."""
    reversed_path = [(ROWS - 1, COLS - 1)]
    current = (ROWS - 1, COLS - 1)
    while current != (0, 0):
        if CHOICE[current] == "right":
            current = (current[0], current[1] - 1)
        else:
            current = (current[0] - 1, current[1])
        reversed_path.append(current)
    forward = []
    index = len(reversed_path) - 1
    while index >= 0:
        forward.append(reversed_path[index])
        index = index - 1
    return forward


PATH = backtrack_path()


def fill_order() -> "list[tuple[int, int]]":
    """Source, then the top row, then the left column, then interior nodes."""
    order = [(0, 0)]
    col = 1
    while col < COLS:
        order.append((0, col))
        col = col + 1
    row = 1
    while row < ROWS:
        order.append((row, 0))
        row = row + 1
    row = 1
    while row < ROWS:
        col = 1
        while col < COLS:
            order.append((row, col))
            col = col + 1
        row = row + 1
    return order


FILL_ORDER = fill_order()


def edge_weight(source: "tuple[int, int]", target: "tuple[int, int]") -> int:
    """Weight of the lattice edge between two adjacent nodes."""
    if target[0] == source[0]:
        return RIGHT_WEIGHTS[source[0]][source[1]]
    return DOWN_WEIGHTS[source[0]][source[1]]


def predecessor(node: "tuple[int, int]", direction: str) -> "tuple[int, int]":
    """The node an incoming edge of the given direction comes from."""
    if direction == "right":
        return (node[0], node[1] - 1)
    return (node[0] - 1, node[1])


def verify() -> "list[str]":
    """Assert the DP, the path, and the drawing geometry. Returns check lines."""
    lines = []

    assert len(RIGHT_WEIGHTS) == ROWS, "RIGHT_WEIGHTS must have one row per node row"
    for row_weights in RIGHT_WEIGHTS:
        assert len(row_weights) == COLS - 1, "each row needs COLS-1 rightward edges"
    assert len(DOWN_WEIGHTS) == ROWS - 1, "DOWN_WEIGHTS must have ROWS-1 rows"
    for row_weights in DOWN_WEIGHTS:
        assert len(row_weights) == COLS, "each row needs COLS downward edges"
    edge_total = ROWS * (COLS - 1) + (ROWS - 1) * COLS
    lines.append("%d nodes, %d weighted edges, weight table well formed"
                 % (ROWS * COLS, edge_total))

    row = 0
    while row < ROWS:
        col = 0
        while col < COLS:
            assert VALUES[row][col] == PUBLISHED_VALUES[row][col], (
                "node (%d,%d) computed %d but the book prints %d"
                % (row, col, VALUES[row][col], PUBLISHED_VALUES[row][col]))
            col = col + 1
        row = row + 1
    lines.append("all 25 recomputed scores match the published figures")

    assert PATH == PUBLISHED_PATH, (
        "backtracked path %s does not match the published red path %s"
        % (PATH, PUBLISHED_PATH))
    lines.append("backtracked path matches the published red path exactly")

    running = 0
    index = 0
    while index < len(PATH) - 1:
        source = PATH[index]
        target = PATH[index + 1]
        step_row = target[0] - source[0]
        step_col = target[1] - source[1]
        assert (step_row, step_col) in [(0, 1), (1, 0)], (
            "path step %s -> %s is not a single right or down move" % (source, target))
        running = running + edge_weight(source, target)
        assert VALUES[target[0]][target[1]] == running, (
            "path prefix sum %d disagrees with node score %d at %s"
            % (running, VALUES[target[0]][target[1]], target))
        index = index + 1
    assert running == VALUES[ROWS - 1][COLS - 1], "path weight is not the sink score"
    lines.append("path weights sum to %d, the sink score, and every prefix agrees"
                 % running)

    row = 0
    while row < ROWS:
        col = 0
        while col < COLS:
            if (row, col) == (0, 0):
                assert (row, col) not in CHOICE, "the source must have no incoming edge"
            else:
                assert (row, col) in CHOICE, "node (%d,%d) has no winning edge" % (row, col)
                direction = CHOICE[(row, col)]
                came_from = predecessor((row, col), direction)
                assert 0 <= came_from[0] < ROWS and 0 <= came_from[1] < COLS, (
                    "node (%d,%d) points off the grid" % (row, col))
                expected = VALUES[came_from[0]][came_from[1]] + edge_weight(
                    came_from, (row, col))
                assert expected == VALUES[row][col], (
                    "winning edge into (%d,%d) does not reproduce its score" % (row, col))
            col = col + 1
        row = row + 1
    lines.append("every non-source node has exactly one winning edge that reproduces it")

    assert len(FILL_ORDER) == ROWS * COLS, "fill order must cover every node once"
    assert len(set(FILL_ORDER)) == len(FILL_ORDER), "fill order repeats a node"
    computed = set()
    for node in FILL_ORDER:
        if node != (0, 0):
            direction = CHOICE[node]
            assert predecessor(node, direction) in computed, (
                "node %s is filled before its predecessor" % (node,))
            if node[0] > 0 and node[1] > 0:
                assert (node[0], node[1] - 1) in computed, "left neighbor not ready"
                assert (node[0] - 1, node[1]) in computed, "upper neighbor not ready"
        computed.add(node)
    lines.append("fill order never computes a node before its predecessors")

    span_x = node_xy(0, COLS - 1)[0] - node_xy(0, 0)[0]
    span_y = node_xy(0, 0)[1] - node_xy(ROWS - 1, 0)[1]
    assert SPACING > 2 * NODE_RADIUS + 0.4, "nodes too close for a weight label"
    assert node_xy(0, 0)[0] - NODE_RADIUS - WEIGHT_OFFSET > 0.3, "grid runs off the left"
    assert node_xy(0, COLS - 1)[0] + NODE_RADIUS < 9.7, "grid runs off the right"
    assert node_xy(0, 0)[1] + NODE_RADIUS < COUNTER_Y - 0.15, "grid collides with counter"
    left_margin = node_xy(0, 0)[0] - NODE_RADIUS
    right_margin = 10.0 - (node_xy(0, COLS - 1)[0] + NODE_RADIUS)
    assert abs(left_margin - right_margin) < 0.01, "grid is not horizontally centered"
    lines.append("grid spans %.2f x %.2f in, centered and clear of the caption"
                 % (span_x, span_y))

    return lines


def draw_node(axis: "plt.Axes", row: int, col: int, label: str,
              ring_color: str) -> None:
    """Shadowed gray lattice node with its score, per the weighted-DAG style."""
    x, y = node_xy(row, col)
    shadow = patches.Circle((x + 0.035, y - 0.035), NODE_RADIUS, facecolor=SHADOW,
                            edgecolor="none", zorder=4)
    axis.add_patch(shadow)
    if ring_color == "":
        outline = NODE_EDGE
        width = 1.9
    else:
        outline = ring_color
        width = 3.0
    circle = patches.Circle((x, y), NODE_RADIUS, facecolor=NODE_FILL,
                            edgecolor=outline, linewidth=width, zorder=5)
    axis.add_patch(circle)
    if label != "":
        axis.text(x, y, label, fontsize=13.5, color=TEXT_DARK, ha="center",
                  va="center", fontproperties=OPTIMA_BOLD, zorder=6)


def draw_edge(axis: "plt.Axes", source: "tuple[int, int]", target: "tuple[int, int]",
              color: str, width: float) -> None:
    """One lattice edge, trimmed short of both nodes, with its weight beside it."""
    ax, ay = node_xy(source[0], source[1])
    bx, by = node_xy(target[0], target[1])
    length = math.hypot(bx - ax, by - ay)
    unit_x = (bx - ax) / length
    unit_y = (by - ay) / length
    trim = NODE_RADIUS + EDGE_GAP
    start = (ax + unit_x * trim, ay + unit_y * trim)
    end = (bx - unit_x * trim, by - unit_y * trim)
    if color == FADED_GRAY:
        zorder = 2
    else:
        zorder = 3
    arrow = patches.FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, linewidth=width,
        color=color, zorder=zorder, joinstyle="round", capstyle="round")
    axis.add_patch(arrow)
    mid_x = (ax + bx) / 2
    mid_y = (ay + by) / 2
    if target[0] == source[0]:
        label_x = mid_x
        label_y = mid_y + WEIGHT_OFFSET
    else:
        label_x = mid_x - WEIGHT_OFFSET
        label_y = mid_y
    axis.text(label_x, label_y, str(edge_weight(source, target)), fontsize=11.5,
              color=weight_text_color(color), ha="center", va="center",
              fontproperties=OPTIMA, zorder=6)


def weight_text_color(edge_color: str) -> str:
    """Edge weights follow their edge when it is emphasized or discarded."""
    if edge_color == EDGE_GRAY or edge_color == NODE_EDGE:
        return TEXT_DARK
    return edge_color


def term_text(target: "tuple[int, int]", direction: str) -> str:
    """The book's wording for one branch of the recurrence."""
    came_from = predecessor(target, direction)
    if direction == "down":
        kind = "vertical"
    else:
        kind = "horizontal"
    return (r"$s_{%d,%d}$ + weight of the %s edge from (%d, %d) to (%d, %d)"
            % (came_from[0], came_from[1], kind, came_from[0], came_from[1],
               target[0], target[1]))


def term_line(target: "tuple[int, int]", direction: str, color: str) -> dict:
    """One recurrence branch plus its two aligned numeric columns."""
    came_from = predecessor(target, direction)
    base = VALUES[came_from[0]][came_from[1]]
    weight = edge_weight(came_from, target)
    return {
        "text": term_text(target, direction),
        "col1": "= %d + %d" % (base, weight),
        "col2": "= %d" % (base + weight),
        "color": color,
    }


def draw_block(axis: "plt.Axes", block: dict) -> None:
    """Render the recurrence block beneath the grid, in the book's layout."""
    lines = block["lines"]
    if block["prefix"] != "":
        axis.text(PREFIX_RIGHT_X, BLOCK_MIDDLE_Y, block["prefix"],
                  fontsize=PREFIX_SIZE, color=TEXT_DARK, ha="right", va="center",
                  fontproperties=OPTIMA)
    if block["brace"]:
        axis.text(BRACE_X, BLOCK_MIDDLE_Y, "{", fontsize=38, color=TEXT_DARK,
                  ha="center", va="center", fontproperties=OPTIMA)
    if len(lines) == 1:
        positions = [BLOCK_MIDDLE_Y]
    else:
        positions = BLOCK_LINE_Y
    index = 0
    while index < len(lines):
        line = lines[index]
        y = positions[index]
        axis.text(TERM_X, y, line["text"], fontsize=TERM_SIZE, color=line["color"],
                  ha="left", va="center", fontproperties=OPTIMA)
        axis.text(COL1_X, y, line["col1"], fontsize=COLUMN_SIZE, color=line["color"],
                  ha="left", va="center", fontproperties=OPTIMA)
        axis.text(COL2_X, y, line["col2"], fontsize=COLUMN_SIZE, color=line["color"],
                  ha="left", va="center", fontproperties=OPTIMA)
        index = index + 1


def draw_frame(spec: dict, output_path: str) -> None:
    """Render one frame from a state dict."""
    figure, axis = plt.subplots(figsize=(10, 7.5), dpi=RENDER_DPI)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 7.5)
    axis.set_aspect("equal")
    axis.axis("off")

    axis.text(5.0, CAPTION_Y, spec["caption"], fontsize=16.5, color=TEXT_DARK,
              ha="center", va="center", fontproperties=OPTIMA_ITALIC)
    if spec["counter"] != "":
        axis.text(5.0, COUNTER_Y, spec["counter"], fontsize=13.5,
                  color=spec["counter_color"], ha="center", va="center",
                  fontproperties=OPTIMA)

    styles = spec["edge_styles"]
    row = 0
    while row < ROWS:
        col = 0
        while col < COLS:
            if col + 1 < COLS:
                key = ((row, col), (row, col + 1))
                style = styles.get(key, (EDGE_GRAY, 1.5))
                draw_edge(axis, key[0], key[1], style[0], style[1])
            if row + 1 < ROWS:
                key = ((row, col), (row + 1, col))
                style = styles.get(key, (EDGE_GRAY, 1.5))
                draw_edge(axis, key[0], key[1], style[0], style[1])
            col = col + 1
        row = row + 1

    labels = spec["node_labels"]
    rings = spec["node_rings"]
    row = 0
    while row < ROWS:
        col = 0
        while col < COLS:
            draw_node(axis, row, col, labels.get((row, col), ""),
                      rings.get((row, col), ""))
            col = col + 1
        row = row + 1

    draw_block(axis, spec["block"])
    figure.savefig(output_path, facecolor="white")
    plt.close(figure)


def reading_duration(caption: str, already_seen: bool) -> int:
    """Still-frame time a caption needs to be readable, from its length."""
    milliseconds = READING_BASE_MS + READING_PER_CHAR_MS * len(caption)
    if milliseconds < READING_MIN_MS:
        milliseconds = READING_MIN_MS
    if milliseconds > READING_MAX_MS:
        milliseconds = READING_MAX_MS
    if already_seen:
        milliseconds = int(milliseconds * REPEAT_CAPTION_FACTOR)
    return int(round(milliseconds / 10.0) * 10)


def empty_block() -> dict:
    """A recurrence block with nothing in it."""
    return {"prefix": "", "brace": False, "lines": []}


def base_spec(caption: str, counter: str, counter_color: str, duration: int) -> dict:
    """An empty frame state with nothing emphasized yet."""
    return {
        "caption": caption,
        "counter": counter,
        "counter_color": counter_color,
        "edge_styles": {},
        "node_labels": {},
        "node_rings": {},
        "block": empty_block(),
        "duration_ms": duration,
    }


def labels_for(computed: "list[tuple[int, int]]") -> dict:
    """Score labels for the nodes computed so far."""
    labels = {}
    for node in computed:
        labels[node] = str(VALUES[node[0]][node[1]])
    return labels


def styles_for(computed: "list[tuple[int, int]]", faded: set) -> dict:
    """Winning edges dark, discarded edges light gray."""
    styles = {}
    for edge in faded:
        styles[edge] = (FADED_GRAY, 1.3)
    for node in computed:
        if node == (0, 0):
            continue
        came_from = predecessor(node, CHOICE[node])
        styles[(came_from, node)] = (NODE_EDGE, 2.9)
    return styles


def losing_edges() -> set:
    """Every edge that no node chose."""
    winners = set()
    losers = set()
    row = 0
    while row < ROWS:
        col = 0
        while col < COLS:
            if (row, col) != (0, 0):
                came_from = predecessor((row, col), CHOICE[(row, col)])
                winners.add((came_from, (row, col)))
            col = col + 1
        row = row + 1
    row = 0
    while row < ROWS:
        col = 0
        while col < COLS:
            if col + 1 < COLS:
                key = ((row, col), (row, col + 1))
                if key not in winners:
                    losers.add(key)
            if row + 1 < ROWS:
                key = ((row, col), (row + 1, col))
                if key not in winners:
                    losers.add(key)
            col = col + 1
        row = row + 1
    return losers


LOSING_EDGES = losing_edges()


def build_fill_specs() -> "list[dict]":
    """The network filling in node by node, each step showing the recurrence."""
    specs = []
    computed = []
    faded = set()
    seen_captions = set()
    interior_index = 0

    intro = "Every node's score is the best of the ways to reach it."
    spec = base_spec(intro, "0 of 25 nodes scored", TEXT_DARK,
                     reading_duration(intro, False))
    spec["block"] = {
        "prefix": r"$s_{i,j}$ = max",
        "brace": True,
        "lines": [
            {"text": r"$s_{i-1,j}$ + weight of the vertical edge into ($i$, $j$)",
             "col1": "", "col2": "", "color": GREEN},
            {"text": r"$s_{i,j-1}$ + weight of the horizontal edge into ($i$, $j$)",
             "col1": "", "col2": "", "color": BLUE},
        ],
    }
    specs.append(spec)
    seen_captions.add(intro)

    node_index = 0
    while node_index < len(FILL_ORDER):
        node = FILL_ORDER[node_index]
        row = node[0]
        col = node[1]

        if node == (0, 0):
            caption = "The source scores 0: no edges have been taken yet."
            step = base_spec(caption, "1 of 25 nodes scored", TEXT_DARK,
                             reading_duration(caption, False))
            computed.append(node)
            step["node_labels"] = labels_for(computed)
            step["edge_styles"] = styles_for(computed, faded)
            step["node_rings"][node] = TEXT_DARK
            step["block"] = {"prefix": r"$s_{0,0}$ = 0", "brace": False, "lines": []}
            specs.append(step)
            seen_captions.add(caption)
            node_index = node_index + 1
            continue

        if row == 0 or col == 0:
            caption = "Along the top row and left column there is only one way in."
            if row == 0:
                direction = "right"
                color = BLUE
            else:
                direction = "down"
                color = GREEN
            computed.append(node)
            line = term_line(node, direction, color)
            line["col2"] = "= %d" % VALUES[row][col]
            if caption in seen_captions:
                duration = INIT_MS
            else:
                duration = reading_duration(caption, False)
            step = base_spec(caption, "%d of 25 nodes scored" % len(computed),
                             TEXT_DARK, duration)
            step["node_labels"] = labels_for(computed)
            step["edge_styles"] = styles_for(computed, faded)
            step["node_rings"][node] = color
            step["block"] = {
                "prefix": r"$s_{%d,%d}$ =" % (row, col),
                "brace": False,
                "lines": [line],
            }
            specs.append(step)
            seen_captions.add(caption)
            node_index = node_index + 1
            continue

        from_left = VALUES[row][col - 1] + RIGHT_WEIGHTS[row][col - 1]
        from_above = VALUES[row - 1][col] + DOWN_WEIGHTS[row - 1][col]
        left_edge = ((row, col - 1), (row, col))
        above_edge = ((row - 1, col), (row, col))
        tie = from_left == from_above
        if tie:
            caption = "Both ways in give the same score, so break the tie horizontally."
        else:
            caption = "Compare the two ways in, then keep the larger."

        block = {
            "prefix": r"$s_{%d,%d}$ = max" % (row, col),
            "brace": True,
            "lines": [term_line(node, "down", GREEN), term_line(node, "right", BLUE)],
        }

        consider = base_spec(
            caption, "%d of 25 nodes scored" % len(computed), TEXT_DARK,
            scheduled_duration(interior_index, CONSIDER_SCHEDULE, CONSIDER_MS_LATE))
        consider["node_labels"] = labels_for(computed)
        consider["edge_styles"] = styles_for(computed, faded)
        consider["edge_styles"][above_edge] = (GREEN, 3.2)
        consider["edge_styles"][left_edge] = (BLUE, 3.2)
        consider["node_rings"][node] = TEXT_DARK
        consider["block"] = block
        specs.append(consider)

        if CHOICE[node] == "right":
            winner_color = BLUE
            winner_edge = left_edge
            loser_edge = above_edge
            loser_position = 0
        else:
            winner_color = GREEN
            winner_edge = above_edge
            loser_edge = left_edge
            loser_position = 1

        computed.append(node)
        faded.add(loser_edge)
        commit = base_spec(
            caption, "%d of 25 nodes scored" % len(computed), TEXT_DARK,
            scheduled_duration(interior_index, COMMIT_SCHEDULE, COMMIT_MS_LATE))
        commit["node_labels"] = labels_for(computed)
        commit["edge_styles"] = styles_for(computed, faded)
        commit["edge_styles"][winner_edge] = (winner_color, 3.2)
        commit["node_rings"][node] = winner_color
        settled = {"prefix": block["prefix"], "brace": True, "lines": []}
        line_index = 0
        while line_index < 2:
            line = dict(block["lines"][line_index])
            if line_index == loser_position:
                line["color"] = FADED_GRAY
            settled["lines"].append(line)
            line_index = line_index + 1
        commit["block"] = settled
        specs.append(commit)

        seen_captions.add(caption)
        interior_index = interior_index + 1
        node_index = node_index + 1

    outro = "Every node scored, and the sink holds the best possible weight."
    final = base_spec(outro, "25 of 25 nodes scored", TEXT_DARK, FINAL_HOLD_MS)
    final["node_labels"] = labels_for(computed)
    final["edge_styles"] = styles_for(computed, faded)
    final["node_rings"][(ROWS - 1, COLS - 1)] = RED
    final["block"] = {
        "prefix": r"$s_{%d,%d}$ = %d" % (ROWS - 1, COLS - 1, VALUES[ROWS - 1][COLS - 1]),
        "brace": False,
        "lines": [],
    }
    specs.append(final)
    return specs


def build_backtrack_specs() -> "list[dict]":
    """Following winning edges back from the sink. The picture carries it."""
    specs = []
    all_labels = labels_for(FILL_ORDER)
    all_computed = list(FILL_ORDER)
    sink = (ROWS - 1, COLS - 1)

    intro = "Each node kept one edge: the one it scored best from."
    spec = base_spec(intro, "", TEXT_DARK, reading_duration(intro, False))
    spec["node_labels"] = all_labels
    spec["edge_styles"] = styles_for(all_computed, LOSING_EDGES)
    spec["node_rings"][sink] = RED
    specs.append(spec)

    traced = []
    index = len(PATH) - 1
    while index > 0:
        target = PATH[index]
        source = PATH[index - 1]
        traced.append((source, target))
        if index == len(PATH) - 1:
            caption = "Start at the sink and step back along the edge it kept."
        else:
            caption = "Keep stepping back, one kept edge at a time."
        step = base_spec(
            caption, "", RED,
            scheduled_duration(len(traced) - 1, BACKTRACK_SCHEDULE, BACKTRACK_MS_LATE))
        step["node_labels"] = all_labels
        step["edge_styles"] = styles_for(all_computed, LOSING_EDGES)
        for edge in traced:
            step["edge_styles"][edge] = (RED, 3.4)
        step["node_rings"][target] = RED
        step["node_rings"][source] = RED
        specs.append(step)
        index = index - 1

    pieces = []
    step_index = 0
    while step_index < len(PATH) - 1:
        pieces.append(str(edge_weight(PATH[step_index], PATH[step_index + 1])))
        step_index = step_index + 1
    outro = ("The longest path: " + " + ".join(pieces)
             + " = %d, the score waiting at the sink." % VALUES[sink[0]][sink[1]])
    final = base_spec(outro, "", RED, FINAL_HOLD_MS)
    final["node_labels"] = all_labels
    final["edge_styles"] = styles_for(all_computed, LOSING_EDGES)
    for edge in traced:
        final["edge_styles"][edge] = (RED, 3.4)
    for node in PATH:
        final["node_rings"][node] = RED
    specs.append(final)
    return specs


def verify_layout() -> "list[str]":
    """Measure the widest recurrence block and assert its columns do not collide."""
    lines = []
    figure, axis = plt.subplots(figsize=(10, 7.5), dpi=RENDER_DPI)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 7.5)
    axis.set_aspect("equal")
    axis.axis("off")
    widest = {
        "prefix": r"$s_{4,4}$ = max",
        "brace": True,
        "lines": [term_line((4, 4), "down", GREEN), term_line((4, 4), "right", BLUE)],
    }
    draw_block(axis, widest)
    figure.canvas.draw()
    transform = axis.transData.inverted()
    boxes = []
    for artist in axis.texts:
        window = artist.get_window_extent(figure.canvas.get_renderer())
        corners = transform.transform([[window.x0, window.y0], [window.x1, window.y1]])
        boxes.append((artist.get_text(), corners[0][0], corners[1][0],
                      corners[0][1], corners[1][1]))
    plt.close(figure)
    index = 0
    while index < len(boxes):
        other = index + 1
        while other < len(boxes):
            a = boxes[index]
            b = boxes[other]
            overlap_x = a[1] < b[2] and b[1] < a[2]
            overlap_y = a[3] < b[4] and b[3] < a[4]
            assert not (overlap_x and overlap_y), (
                "recurrence text collides: %r and %r" % (a[0][:28], b[0][:28]))
            other = other + 1
        index = index + 1
    right_edge = 0.0
    for box in boxes:
        if box[2] > right_edge:
            right_edge = box[2]
    assert right_edge < 9.7, "recurrence block runs off the canvas at %.2f" % right_edge
    lowest = 7.5
    for box in boxes:
        if box[3] < lowest:
            lowest = box[3]
    assert lowest > 0.15, "recurrence block runs off the bottom at %.2f" % lowest
    grid_bottom = node_xy(ROWS - 1, 0)[1] - NODE_RADIUS
    highest = 0.0
    for box in boxes:
        if box[4] > highest:
            highest = box[4]
    assert highest < grid_bottom - 0.1, (
        "recurrence block at %.2f overlaps the grid bottom at %.2f"
        % (highest, grid_bottom))
    lines.append("widest recurrence block fits, no collisions, clear of the grid")

    assert len(LOSING_EDGES) == 16, "expected 16 discarded edges, got %d" % len(LOSING_EDGES)
    total_edges = ROWS * (COLS - 1) + (ROWS - 1) * COLS
    assert len(LOSING_EDGES) + (ROWS * COLS - 1) == total_edges, (
        "winners plus losers must account for every edge")
    lines.append("24 kept edges plus 16 discarded edges account for all 40")

    fill_specs = build_fill_specs()
    back_specs = build_backtrack_specs()
    fill_final = fill_specs[len(fill_specs) - 1]["edge_styles"]
    back_first = back_specs[0]["edge_styles"]
    fill_faded = set()
    for edge in fill_final:
        if fill_final[edge][0] == FADED_GRAY:
            fill_faded.add(edge)
    back_faded = set()
    for edge in back_first:
        if back_first[edge][0] == FADED_GRAY:
            back_faded.add(edge)
    assert fill_faded == back_faded, "the two animations disagree about discarded edges"
    lines.append("the fill's last frame and the backtrack's first frame agree exactly")
    return lines


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: manhattan_dp.py fill|backtrack OUTPUT.gif")
        return
    which = sys.argv[1]
    output_path = sys.argv[2]

    print("Structural checks:")
    for line in verify():
        print("  ok: " + line)
    for line in verify_layout():
        print("  ok: " + line)

    if which == "fill":
        specs = build_fill_specs()
    elif which == "backtrack":
        globals()["GRID_TOP"] = GRID_TOP_BACKTRACK
        specs = build_backtrack_specs()
    else:
        print("first argument must be fill or backtrack")
        return

    durations = []
    for spec in specs:
        durations.append(spec["duration_ms"])
    print("Rendering %d frames at %dx%d, %.1f s of playback..."
          % (len(specs), OUTPUT_WIDTH, OUTPUT_HEIGHT, sum(durations) / 1000.0))
    directory = tempfile.mkdtemp(prefix="manhattan_" + which + "_")
    frame_paths = []
    index = 0
    while index < len(specs):
        path = os.path.join(directory, "frame_%04d.png" % index)
        draw_frame(specs[index], path)
        frame_paths.append(path)
        index = index + 1

    assemble_gif(frame_paths, output_path, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT,
                 frame_durations=durations)
    print("Saved " + output_path)


if __name__ == "__main__":
    main()
