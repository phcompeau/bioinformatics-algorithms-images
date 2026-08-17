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

GRID_LEFT = 0.95
GRID_TOP = 5.95
SPACING = 1.32
NODE_RADIUS = 0.30
EDGE_GAP = 0.07
WEIGHT_OFFSET = 0.23

PANEL_X = 7.15
CAPTION_Y = 7.05
COUNTER_Y = 6.55

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
    assert node_xy(0, 0)[0] - NODE_RADIUS > 0.3, "grid runs off the left edge"
    assert node_xy(ROWS - 1, 0)[1] - NODE_RADIUS > 0.3, "grid runs off the bottom"
    assert node_xy(0, COLS - 1)[0] + NODE_RADIUS < PANEL_X - 0.3, "grid collides with panel"
    lines.append("grid spans %.2f x %.2f in, clear of the canvas edges and the panel"
                 % (span_x, span_y))

    return lines


def draw_node(axis: "plt.Axes", row: int, col: int, label: str,
              ring_color: str, dim: bool) -> None:
    """Shadowed gray lattice node with its score, per the weighted-DAG style."""
    x, y = node_xy(row, col)
    shadow = patches.Circle((x + 0.04, y - 0.04), NODE_RADIUS, facecolor=SHADOW,
                            edgecolor="none", zorder=4)
    axis.add_patch(shadow)
    if ring_color == "":
        outline = NODE_EDGE
        width = 2.0
    else:
        outline = ring_color
        width = 3.0
    if dim:
        outline = FADED_GRAY
        width = 1.4
    circle = patches.Circle((x, y), NODE_RADIUS, facecolor=NODE_FILL,
                            edgecolor=outline, linewidth=width, zorder=5)
    axis.add_patch(circle)
    if label != "":
        axis.text(x, y, label, fontsize=15, color=TEXT_DARK, ha="center",
                  va="center", fontproperties=OPTIMA_BOLD, zorder=6)


def draw_edge(axis: "plt.Axes", source: "tuple[int, int]", target: "tuple[int, int]",
              color: str, width: float, show_weight: bool, weight_color: str) -> None:
    """One lattice edge, trimmed short of both nodes, with its weight beside it."""
    ax, ay = node_xy(source[0], source[1])
    bx, by = node_xy(target[0], target[1])
    length = math.hypot(bx - ax, by - ay)
    unit_x = (bx - ax) / length
    unit_y = (by - ay) / length
    trim = NODE_RADIUS + EDGE_GAP
    start = (ax + unit_x * trim, ay + unit_y * trim)
    end = (bx - unit_x * trim, by - unit_y * trim)
    arrow = patches.FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=width,
        color=color, zorder=3, joinstyle="round", capstyle="round")
    axis.add_patch(arrow)
    if not show_weight:
        return
    mid_x = (ax + bx) / 2
    mid_y = (ay + by) / 2
    if target[0] == source[0]:
        label_x = mid_x
        label_y = mid_y + WEIGHT_OFFSET
    else:
        label_x = mid_x - WEIGHT_OFFSET
        label_y = mid_y
    axis.text(label_x, label_y, str(edge_weight(source, target)), fontsize=13,
              color=weight_color, ha="center", va="center", fontproperties=OPTIMA,
              zorder=6)


def draw_frame(spec: dict, output_path: str) -> None:
    """Render one frame from a state dict."""
    figure, axis = plt.subplots(figsize=(10, 7.5), dpi=100)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 7.5)
    axis.set_aspect("equal")
    axis.axis("off")

    axis.text(5.0, CAPTION_Y, spec["caption"], fontsize=17, color=TEXT_DARK,
              ha="center", va="center", fontproperties=OPTIMA_ITALIC)
    axis.text(5.0, COUNTER_Y, spec["counter"], fontsize=14,
              color=spec["counter_color"], ha="center", va="center",
              fontproperties=OPTIMA)

    edge_styles = spec["edge_styles"]
    hidden = spec["hidden_edges"]
    row = 0
    while row < ROWS:
        col = 0
        while col < COLS:
            if col + 1 < COLS:
                key = ((row, col), (row, col + 1))
                if key not in hidden:
                    style = edge_styles.get(key, (EDGE_GRAY, 1.6))
                    draw_edge(axis, key[0], key[1], style[0], style[1],
                              spec["show_weights"], weight_text_color(style[0]))
            if row + 1 < ROWS:
                key = ((row, col), (row + 1, col))
                if key not in hidden:
                    style = edge_styles.get(key, (EDGE_GRAY, 1.6))
                    draw_edge(axis, key[0], key[1], style[0], style[1],
                              spec["show_weights"], weight_text_color(style[0]))
            col = col + 1
        row = row + 1

    labels = spec["node_labels"]
    rings = spec["node_rings"]
    row = 0
    while row < ROWS:
        col = 0
        while col < COLS:
            label = labels.get((row, col), "")
            ring = rings.get((row, col), "")
            dim = (row, col) in spec["dim_nodes"]
            draw_node(axis, row, col, label, ring, dim)
            col = col + 1
        row = row + 1

    line_index = 0
    while line_index < len(spec["panel"]):
        text, color, size = spec["panel"][line_index]
        axis.text(PANEL_X, 5.35 - line_index * 0.46, text, fontsize=size,
                  color=color, ha="left", va="center", fontproperties=OPTIMA)
        line_index = line_index + 1

    figure.savefig(output_path, facecolor="white")
    plt.close(figure)


def weight_text_color(edge_color: str) -> str:
    """Edge weights follow their edge when it is emphasized."""
    if edge_color == EDGE_GRAY:
        return TEXT_DARK
    if edge_color == FADED_GRAY:
        return FADED_GRAY
    return edge_color


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


def base_spec(caption: str, counter: str, counter_color: str, duration: int) -> dict:
    """An empty frame state with nothing emphasized yet."""
    return {
        "caption": caption,
        "counter": counter,
        "counter_color": counter_color,
        "edge_styles": {},
        "hidden_edges": set(),
        "node_labels": {},
        "node_rings": {},
        "dim_nodes": set(),
        "panel": [],
        "show_weights": True,
        "duration_ms": duration,
    }


def labels_for(computed: "list[tuple[int, int]]") -> dict:
    """Score labels for the nodes computed so far."""
    labels = {}
    for node in computed:
        labels[node] = str(VALUES[node[0]][node[1]])
    return labels


def winning_styles(computed: "list[tuple[int, int]]") -> dict:
    """Thicken the winning incoming edge of every node computed so far."""
    styles = {}
    for node in computed:
        if node == (0, 0):
            continue
        came_from = predecessor(node, CHOICE[node])
        styles[(came_from, node)] = (NODE_EDGE, 3.0)
    return styles


def build_fill_specs() -> "list[dict]":
    """The network filling in node by node."""
    specs = []
    computed = []
    seen_captions = set()
    interior_index = 0

    intro = "Every node's score is the best of the ways to reach it."
    spec = base_spec(intro, "0 of 25 nodes scored", TEXT_DARK,
                     reading_duration(intro, False))
    spec["panel"] = [
        ("The Manhattan network,", TEXT_DARK, 14),
        ("with a weight on every edge.", TEXT_DARK, 14),
        ("", TEXT_DARK, 14),
        ("Find the highest-scoring", TEXT_DARK, 14),
        ("path from corner to corner.", TEXT_DARK, 14),
    ]
    specs.append(spec)
    seen_captions.add(intro)

    node_index = 0
    while node_index < len(FILL_ORDER):
        node = FILL_ORDER[node_index]
        row = node[0]
        col = node[1]
        interior = row > 0 and col > 0

        if node == (0, 0):
            caption = "The source scores 0: no edges have been taken yet."
        elif not interior:
            caption = "Along the top row and left column there is only one way in."
        else:
            caption = "Compare the two ways in, then keep the larger."

        if interior:
            from_left = VALUES[row][col - 1] + RIGHT_WEIGHTS[row][col - 1]
            from_above = VALUES[row - 1][col] + DOWN_WEIGHTS[row - 1][col]
            left_edge = ((row, col - 1), (row, col))
            above_edge = ((row - 1, col), (row, col))

            consider = base_spec(
                caption, "%d of 25 nodes scored" % len(computed), TEXT_DARK,
                scheduled_duration(interior_index, CONSIDER_SCHEDULE, CONSIDER_MS_LATE))
            consider["node_labels"] = labels_for(computed)
            consider["edge_styles"] = winning_styles(computed)
            consider["edge_styles"][left_edge] = (BLUE, 3.4)
            consider["edge_styles"][above_edge] = (GREEN, 3.4)
            consider["node_rings"][node] = TEXT_DARK
            consider["panel"] = [
                ("from the left", BLUE, 14),
                ("   %d + %d = %d" % (VALUES[row][col - 1],
                                      RIGHT_WEIGHTS[row][col - 1], from_left), BLUE, 15),
                ("from above", GREEN, 14),
                ("   %d + %d = %d" % (VALUES[row - 1][col],
                                      DOWN_WEIGHTS[row - 1][col], from_above), GREEN, 15),
            ]
            specs.append(consider)

            if CHOICE[node] == "right":
                winner_color = BLUE
                winner_edge = left_edge
                loser_edge = above_edge
                winner_value = from_left
                winner_word = "from the left"
            else:
                winner_color = GREEN
                winner_edge = above_edge
                loser_edge = left_edge
                winner_value = from_above
                winner_word = "from above"

            commit = base_spec(
                caption, "%d of 25 nodes scored" % (len(computed) + 1), TEXT_DARK,
                scheduled_duration(interior_index, COMMIT_SCHEDULE, COMMIT_MS_LATE))
            computed.append(node)
            commit["node_labels"] = labels_for(computed)
            commit["edge_styles"] = winning_styles(computed)
            commit["edge_styles"][winner_edge] = (winner_color, 3.4)
            commit["edge_styles"][loser_edge] = (FADED_GRAY, 1.4)
            commit["node_rings"][node] = winner_color
            tie_note = ""
            if from_left == from_above:
                tie_note = "a tie, broken horizontally"
            commit["panel"] = [
                ("from the left", BLUE, 14),
                ("   %d + %d = %d" % (VALUES[row][col - 1],
                                      RIGHT_WEIGHTS[row][col - 1], from_left), BLUE, 15),
                ("from above", GREEN, 14),
                ("   %d + %d = %d" % (VALUES[row - 1][col],
                                      DOWN_WEIGHTS[row - 1][col], from_above), GREEN, 15),
                ("", TEXT_DARK, 14),
                ("larger is %d, %s" % (winner_value, winner_word), winner_color, 15),
            ]
            if tie_note != "":
                commit["panel"].append((tie_note, TEXT_DARK, 13))
            specs.append(commit)
            interior_index = interior_index + 1
        else:
            computed.append(node)
            if caption not in seen_captions:
                hold = base_spec(caption, "%d of 25 nodes scored" % len(computed),
                                 TEXT_DARK, reading_duration(caption, False))
                hold["node_labels"] = labels_for(computed)
                hold["edge_styles"] = winning_styles(computed)
                hold["node_rings"][node] = TEXT_DARK
                specs.append(hold)
                seen_captions.add(caption)
            else:
                step = base_spec(caption, "%d of 25 nodes scored" % len(computed),
                                 TEXT_DARK, INIT_MS)
                step["node_labels"] = labels_for(computed)
                step["edge_styles"] = winning_styles(computed)
                step["node_rings"][node] = TEXT_DARK
                specs.append(step)

        if caption not in seen_captions:
            seen_captions.add(caption)
        node_index = node_index + 1

    outro = "Every node scored. The sink's %d is the best possible." % VALUES[ROWS - 1][COLS - 1]
    final = base_spec(outro, "25 of 25 nodes scored", TEXT_DARK, FINAL_HOLD_MS)
    final["node_labels"] = labels_for(computed)
    final["edge_styles"] = winning_styles(computed)
    final["node_rings"][(ROWS - 1, COLS - 1)] = RED
    final["panel"] = [
        ("Each dark edge is the one", TEXT_DARK, 14),
        ("its node scored best from.", TEXT_DARK, 14),
        ("", TEXT_DARK, 14),
        ("Follow them back from the", TEXT_DARK, 14),
        ("sink to recover the path.", TEXT_DARK, 14),
    ]
    specs.append(final)
    return specs


def hidden_losing_edges() -> set:
    """Every edge that no node chose, so only winning edges remain drawn."""
    winners = set()
    row = 0
    while row < ROWS:
        col = 0
        while col < COLS:
            if (row, col) != (0, 0):
                came_from = predecessor((row, col), CHOICE[(row, col)])
                winners.add((came_from, (row, col)))
            col = col + 1
        row = row + 1
    hidden = set()
    row = 0
    while row < ROWS:
        col = 0
        while col < COLS:
            if col + 1 < COLS:
                key = ((row, col), (row, col + 1))
                if key not in winners:
                    hidden.add(key)
            if row + 1 < ROWS:
                key = ((row, col), (row + 1, col))
                if key not in winners:
                    hidden.add(key)
            col = col + 1
        row = row + 1
    return hidden


def build_backtrack_specs() -> "list[dict]":
    """Following winning edges back from the sink to build the longest path."""
    specs = []
    hidden = hidden_losing_edges()
    all_labels = labels_for(FILL_ORDER)
    sink = (ROWS - 1, COLS - 1)

    intro = "Keep only the winning edges: one into every node."
    spec = base_spec(intro, "score at the sink: %d" % VALUES[sink[0]][sink[1]],
                     TEXT_DARK, reading_duration(intro, False))
    spec["hidden_edges"] = hidden
    spec["node_labels"] = all_labels
    spec["edge_styles"] = winning_styles(FILL_ORDER)
    spec["node_rings"][sink] = RED
    spec["panel"] = [
        ("Every node remembers the", TEXT_DARK, 14),
        ("one edge it scored best from.", TEXT_DARK, 14),
        ("", TEXT_DARK, 14),
        ("Walk those edges backwards", TEXT_DARK, 14),
        ("from the sink.", TEXT_DARK, 14),
    ]
    specs.append(spec)

    traced = []
    index = len(PATH) - 1
    while index > 0:
        target = PATH[index]
        source = PATH[index - 1]
        traced.append((source, target))
        weight = edge_weight(source, target)
        if CHOICE[target] == "right":
            direction_word = "from the left"
        else:
            direction_word = "from above"
        caption = "The sink's score came from here, so step back along that edge."
        if index < len(PATH) - 1:
            caption = "Each score names the edge that produced it. Step back again."
        step = base_spec(
            caption, "score at the sink: %d" % VALUES[sink[0]][sink[1]], RED,
            scheduled_duration(len(traced) - 1, BACKTRACK_SCHEDULE, BACKTRACK_MS_LATE))
        step["hidden_edges"] = hidden
        step["node_labels"] = all_labels
        step["edge_styles"] = winning_styles(FILL_ORDER)
        for edge in traced:
            step["edge_styles"][edge] = (RED, 3.6)
        step["node_rings"][target] = RED
        step["node_rings"][source] = RED
        step["panel"] = [
            ("%d %s" % (VALUES[target[0]][target[1]], direction_word), RED, 15),
            ("   %d + %d = %d" % (VALUES[source[0]][source[1]], weight,
                                  VALUES[target[0]][target[1]]), RED, 15),
            ("", TEXT_DARK, 14),
            ("%d of %d edges traced" % (len(traced), len(PATH) - 1), TEXT_DARK, 14),
        ]
        specs.append(step)
        index = index - 1

    pieces = []
    step_index = 0
    while step_index < len(PATH) - 1:
        pieces.append(str(edge_weight(PATH[step_index], PATH[step_index + 1])))
        step_index = step_index + 1
    sum_text = " + ".join(pieces) + " = %d" % VALUES[sink[0]][sink[1]]

    outro = "The longest path, recovered backwards and read forwards."
    final = base_spec(outro, "score at the sink: %d" % VALUES[sink[0]][sink[1]],
                      RED, FINAL_HOLD_MS)
    final["hidden_edges"] = hidden
    final["node_labels"] = all_labels
    final["edge_styles"] = winning_styles(FILL_ORDER)
    for edge in traced:
        final["edge_styles"][edge] = (RED, 3.6)
    for node in PATH:
        final["node_rings"][node] = RED
    final["panel"] = [
        ("Edge weights along the path:", TEXT_DARK, 14),
        (sum_text, RED, 15),
        ("", TEXT_DARK, 14),
        ("which is exactly the score", TEXT_DARK, 14),
        ("stored at the sink.", TEXT_DARK, 14),
    ]
    specs.append(final)
    return specs


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: manhattan_dp.py fill|backtrack OUTPUT.gif")
        return
    which = sys.argv[1]
    output_path = sys.argv[2]

    print("Structural checks:")
    for line in verify():
        print("  ok: " + line)

    if which == "fill":
        specs = build_fill_specs()
    elif which == "backtrack":
        specs = build_backtrack_specs()
    else:
        print("first argument must be fill or backtrack")
        return

    durations = []
    for spec in specs:
        durations.append(spec["duration_ms"])
    print("Rendering %d frames, %.1f s of playback..."
          % (len(specs), sum(durations) / 1000.0))
    directory = tempfile.mkdtemp(prefix="manhattan_" + which + "_")
    frame_paths = []
    index = 0
    while index < len(specs):
        path = os.path.join(directory, "frame_%04d.png" % index)
        draw_frame(specs[index], path)
        frame_paths.append(path)
        index = index + 1

    assemble_gif(frame_paths, output_path, width=800, height=600,
                 frame_durations=durations)
    print("Saved " + output_path)


if __name__ == "__main__":
    main()
