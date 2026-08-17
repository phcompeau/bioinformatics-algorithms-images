"""Animated suffix trie and suffix tree of panamabananas$, in the Rosalind style.

Two animations from Chapter 9, built from one string and one verified structure:

  trie     - the suffix trie grows one suffix at a time. Each insertion threads
             through the nodes that already exist (blue) and then adds the nodes
             that are new (green), ending in a numbered leaf.
  compress - every non-branching path collapses into a single edge, turning the
             suffix trie into the suffix tree. The letters that were separate
             edge labels slide together into one stacked label.

Both structures are built from the text, then asserted against the published
figures images/BWT/suffix_trie.png and images/BWT/suffix_tree.png.

Run:  python3 suffix_tree.py trie OUTPUT.gif
      python3 suffix_tree.py compress OUTPUT.gif
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
ORANGE = "#F46E2B"
EDGE_GRAY = "#595959"
FADED_GRAY = "#D2D2D2"
NODE_FILL = "#DCDCDC"
LEAF_FILL = "#3F3F3F"
SHADOW = "#BFBFBF"
TEXT_DARK = "#262626"

OPTIMA = font_manager.FontProperties(family="Optima")
OPTIMA_ITALIC = font_manager.FontProperties(family="Optima", style="italic")
COURIER = font_manager.FontProperties(family="Courier New", weight="bold")

TEXT = "panamabananas$"

# Transcribed from images/BWT/suffix_tree.png, used as a correctness oracle.
PUBLISHED_TREE = {
    "": {"$": "leaf 13", "a": "a", "bananas$": "leaf 6", "mabananas$": "leaf 4",
         "na": "na", "panamabananas$": "leaf 0", "s$": "leaf 12"},
    "a": {"bananas$": "leaf 5", "mabananas$": "leaf 3", "na": "ana",
          "s$": "leaf 11"},
    "ana": {"mabananas$": "leaf 1", "nas$": "leaf 7", "s$": "leaf 9"},
    "na": {"mabananas$": "leaf 2", "nas$": "leaf 8", "s$": "leaf 10"},
}
PUBLISHED_ROOT_BRANCHES = "$abmnps"

FIGURE_WIDTH = 11.0
FIGURE_HEIGHT = 13.0
RENDER_DPI = 100
OUTPUT_WIDTH = 1100
OUTPUT_HEIGHT = 1300

TRIE_TOP = 12.35
TRIE_GAP = 0.80
LEFT_MARGIN = 0.42
NODE_RADIUS = 0.165
LEAF_RADIUS = 0.185
EDGE_GAP = 0.03
LETTER_OFFSET = 0.17
LETTER_SIZE = 9.5
LEAF_SIZE = 8.5

TREE_LETTER_GAP = 0.30
TREE_EDGE_PAD = 0.26

READOUT_Y = 0.42
READOUT_SIZE = 15.0

TRIE_THREAD_MS = 900
TRIE_LETTER_MS = 220
TRIE_LEAF_MS = 800
TRIE_SETTLE_MS = 360
TRIE_FIRST_MS = 1400
FINAL_HOLD_MS = 4600

COMPRESS_INTRO_MS = 3200
COMPRESS_MARK_MS = 3000
COMPRESS_MOTION_FRAMES = 26
COMPRESS_MOTION_MS = 120


def child_sort_key(character: str) -> str:
    """The book orders root branches $ then alphabetically."""
    if character == "$":
        return " "
    return character


def build_trie() -> "tuple[dict, dict, dict]":
    """children[node][char] = child, leaf_label[node] = suffix start, parent_char."""
    children = {0: {}}
    leaf_label = {}
    incoming = {}
    next_id = 1
    start = 0
    while start < len(TEXT):
        suffix = TEXT[start:]
        current = 0
        position = 0
        while position < len(suffix):
            character = suffix[position]
            if character not in children[current]:
                children[next_id] = {}
                children[current][character] = next_id
                incoming[next_id] = character
                next_id = next_id + 1
            current = children[current][character]
            position = position + 1
        leaf_label[current] = start
        start = start + 1
    return (children, leaf_label, incoming)


CHILDREN, LEAF_LABEL, INCOMING = build_trie()


def sorted_children(node: int) -> "list[int]":
    """Child nodes of a trie node, in the book's left-to-right order."""
    characters = []
    for character in CHILDREN[node]:
        characters.append(character)
    characters.sort(key=child_sort_key)
    result = []
    for character in characters:
        result.append(CHILDREN[node][character])
    return result


def depths() -> dict:
    """Depth of every trie node, the root being 0."""
    result = {0: 0}
    stack = [0]
    while len(stack) > 0:
        node = stack.pop()
        for child in sorted_children(node):
            result[child] = result[node] + 1
            stack.append(child)
    return result


DEPTH = depths()


def parent_map() -> dict:
    """Parent of every trie node except the root."""
    result = {}
    stack = [0]
    while len(stack) > 0:
        node = stack.pop()
        for child in sorted_children(node):
            result[child] = node
            stack.append(child)
    return result


PARENT = parent_map()


def spell(node: int) -> str:
    """The string a trie node spells from the root."""
    letters = []
    walker = node
    while walker != 0:
        letters.append(INCOMING[walker])
        walker = PARENT[walker]
    letters.reverse()
    return "".join(letters)


def leaf_order() -> "list[int]":
    """Trie leaves left to right, by depth-first traversal in book order."""
    order = []
    stack = [0]
    while len(stack) > 0:
        node = stack.pop()
        kids = sorted_children(node)
        if len(kids) == 0:
            order.append(node)
        index = len(kids) - 1
        while index >= 0:
            stack.append(kids[index])
            index = index - 1
    return order


LEAF_ORDER = leaf_order()


def x_positions() -> dict:
    """Leaves take evenly spaced slots; a parent centers over its children."""
    span = FIGURE_WIDTH - 2 * LEFT_MARGIN
    slot = span / (len(LEAF_ORDER) - 1)
    result = {}
    index = 0
    while index < len(LEAF_ORDER):
        result[LEAF_ORDER[index]] = LEFT_MARGIN + index * slot
        index = index + 1
    ordered = []
    for node in DEPTH:
        ordered.append((DEPTH[node], node))
    ordered.sort(reverse=True)
    for depth, node in ordered:
        kids = sorted_children(node)
        if len(kids) == 0:
            continue
        total = 0.0
        for child in kids:
            total = total + result[child]
        result[node] = total / len(kids)
    return result


X = x_positions()


def trie_positions() -> dict:
    """Trie node centers: depth sets y, the slot layout sets x."""
    result = {}
    for node in DEPTH:
        result[node] = (X[node], TRIE_TOP - DEPTH[node] * TRIE_GAP)
    return result


TRIE_XY = trie_positions()


def build_tree() -> "tuple[dict, dict]":
    """Collapse non-branching paths.

    Returns edges[top] = {label: bottom} and owner[node] = (top, bottom, index),
    which says which collapsed edge each trie node's letter belongs to and where
    in that edge's label it sits.
    """
    edges = {}
    owner = {}
    pending = [0]
    while len(pending) > 0:
        top = pending.pop()
        edges[top] = {}
        for child in sorted_children(top):
            label = INCOMING[child]
            owner[child] = (top, None, 0)
            chain = [child]
            walker = child
            while len(CHILDREN[walker]) == 1 and walker not in LEAF_LABEL:
                only = sorted_children(walker)[0]
                label = label + INCOMING[only]
                chain.append(only)
                walker = only
            edges[top][label] = walker
            index = 0
            while index < len(chain):
                owner[chain[index]] = (top, walker, index)
                index = index + 1
            if walker not in LEAF_LABEL:
                pending.append(walker)
    return (edges, owner)


TREE_EDGES, OWNER = build_tree()


def tree_positions() -> dict:
    """Surviving node centers once every chain has collapsed.

    An edge's length is proportional to its label length, so the whole figure
    shrinks vertically. x never changes, which is what makes the collapse read
    as pure compression.
    """
    result = {0: (X[0], TRIE_TOP)}
    pending = [0]
    while len(pending) > 0:
        top = pending.pop()
        for label in TREE_EDGES[top]:
            bottom = TREE_EDGES[top][label]
            drop = len(label) * TREE_LETTER_GAP + TREE_EDGE_PAD
            result[bottom] = (X[bottom], result[top][1] - drop)
            if bottom in TREE_EDGES:
                pending.append(bottom)
    return result


TREE_XY = tree_positions()
SURVIVORS = set(TREE_XY.keys())


def letters_beside(positions: dict) -> dict:
    """Each node's incoming letter, beside the midpoint of its own edge.

    Used for both the trie layout and the collapsed layout, so a letter never
    changes which side of its edge it sits on while the figure compresses.
    """
    result = {}
    for node in INCOMING:
        parent = PARENT[node]
        ax, ay = positions[parent]
        bx, by = positions[node]
        length = math.hypot(bx - ax, by - ay)
        perp_x = -(by - ay) / length
        perp_y = (bx - ax) / length
        mid_x = (ax + bx) / 2
        mid_y = (ay + by) / 2
        result[node] = (mid_x + perp_x * LETTER_OFFSET,
                        mid_y + perp_y * LETTER_OFFSET)
    return result


def collapsed_positions() -> dict:
    """Every node's position once the chains have collapsed.

    A chain keeps its x and stays vertical, exactly as the published tree draws
    it: one diagonal step off the branch node, then a straight run of letters.
    Only y changes, which is why the collapse reads as pure compression.
    """
    result = {}
    for node in TREE_XY:
        result[node] = TREE_XY[node]
    for node in INCOMING:
        if node in SURVIVORS:
            continue
        top, bottom, index = OWNER[node]
        label_length = 0
        for label in TREE_EDGES[top]:
            if TREE_EDGES[top][label] == bottom:
                label_length = len(label)
        remaining = label_length - 1 - index
        result[node] = (X[node], TREE_XY[bottom][1] + remaining * TREE_LETTER_GAP)
    return result


COLLAPSED_XY = collapsed_positions()
LETTER_TRIE = letters_beside(TRIE_XY)
LETTER_TREE = letters_beside(COLLAPSED_XY)


def insertion_plan() -> "list[dict]":
    """Per suffix: the nodes it threads through and the nodes it creates."""
    plan = []
    existing = set([0])
    start = 0
    while start < len(TEXT):
        suffix = TEXT[start:]
        path = []
        current = 0
        position = 0
        while position < len(suffix):
            current = CHILDREN[current][suffix[position]]
            path.append(current)
            position = position + 1
        threaded = []
        created = []
        for node in path:
            if node in existing:
                threaded.append(node)
            else:
                created.append(node)
        for node in path:
            existing.add(node)
        plan.append({
            "start": start,
            "suffix": suffix,
            "threaded": threaded,
            "created": created,
            "leaf": path[len(path) - 1],
        })
        start = start + 1
    return plan


PLAN = insertion_plan()


def verify() -> "list[str]":
    """Assert both structures against the published figures."""
    lines = []

    assert len(LEAF_LABEL) == len(TEXT), "one leaf per suffix"
    distinct = set()
    start = 0
    while start < len(TEXT):
        end = start + 1
        while end <= len(TEXT):
            distinct.add(TEXT[start:end])
            end = end + 1
        start = start + 1
    assert len(CHILDREN) == len(distinct) + 1, "trie nodes must be distinct substrings"
    lines.append("suffix trie: %d nodes, one per distinct substring plus the root"
                 % len(CHILDREN))

    for node in LEAF_LABEL:
        assert spell(node) == TEXT[LEAF_LABEL[node]:], "a leaf misspells its suffix"
    lines.append("every one of the %d leaves spells its own suffix" % len(LEAF_LABEL))

    branches = []
    for character in CHILDREN[0]:
        branches.append(character)
    branches.sort(key=child_sort_key)
    assert "".join(branches) == PUBLISHED_ROOT_BRANCHES, (
        "root branches %s, figure shows %s"
        % ("".join(branches), PUBLISHED_ROOT_BRANCHES))
    lines.append("root branches on %s, matching suffix_trie.png" % "".join(branches))

    rebuilt = {}
    for top in TREE_EDGES:
        key = spell(top)
        rebuilt[key] = {}
        for label in TREE_EDGES[top]:
            bottom = TREE_EDGES[top][label]
            if bottom in LEAF_LABEL:
                rebuilt[key][label] = "leaf %d" % LEAF_LABEL[bottom]
            else:
                rebuilt[key][label] = spell(bottom)
    assert rebuilt == PUBLISHED_TREE, "computed suffix tree disagrees with the figure"
    edge_total = 0
    for top in TREE_EDGES:
        edge_total = edge_total + len(TREE_EDGES[top])
    lines.append("suffix tree: %d edges over %d nodes, matching suffix_tree.png exactly"
                 % (edge_total, edge_total + 1))

    for top in TREE_EDGES:
        for label in TREE_EDGES[top]:
            bottom = TREE_EDGES[top][label]
            assert spell(top) + label == spell(bottom), (
                "collapsed edge %s does not spell its endpoints" % label)
    lines.append("every collapsed edge label spells the gap between its endpoints")

    assert len(SURVIVORS) == edge_total + 1, "survivor count must equal tree nodes"
    for node in LEAF_LABEL:
        assert node in SURVIVORS, "a leaf was collapsed away"
    lines.append("all %d leaves and %d branch nodes survive the collapse"
                 % (len(LEAF_LABEL), len(SURVIVORS) - len(LEAF_LABEL)))

    covered = set()
    for node in INCOMING:
        top, bottom, index = OWNER[node]
        assert bottom is not None, "node %d has no collapsed edge" % node
        covered.add(node)
    assert len(covered) == len(CHILDREN) - 1, "every non-root node needs an owner"
    lines.append("every letter belongs to exactly one collapsed edge")

    for node in LEAF_LABEL:
        letters = len(spell(node))
        expected = TRIE_TOP - letters * TRIE_GAP
        assert abs(TRIE_XY[node][1] - expected) < 1e-9, "trie depth mismatch"
    lines.append("trie leaf depth equals suffix length for all %d leaves"
                 % len(LEAF_LABEL))

    for node in SURVIVORS:
        assert abs(TRIE_XY[node][0] - TREE_XY[node][0]) < 1e-9, (
            "node %d moves horizontally during the collapse" % node)
    lines.append("no surviving node moves horizontally: the collapse is vertical only")

    total = 0
    for step in PLAN:
        total = total + len(step["created"])
    assert total == len(CHILDREN) - 1, "insertions must create every node once"
    lines.append("the %d insertions create all %d non-root nodes exactly once"
                 % (len(PLAN), total))

    lowest = TRIE_TOP
    for node in TRIE_XY:
        if TRIE_XY[node][1] < lowest:
            lowest = TRIE_XY[node][1]
    assert lowest - LEAF_RADIUS > READOUT_Y + 0.28, "trie collides with the readout"
    assert TRIE_TOP + NODE_RADIUS < FIGURE_HEIGHT - 0.2, "trie runs off the top"
    lines.append("trie spans %.2f to %.2f in, clear of the readout line"
                 % (lowest, TRIE_TOP))

    tree_lowest = TRIE_TOP
    for node in TREE_XY:
        if TREE_XY[node][1] < tree_lowest:
            tree_lowest = TREE_XY[node][1]
    lines.append("the tree is %.1fx shorter than the trie (%.2f vs %.2f in tall)"
                 % ((TRIE_TOP - lowest) / (TRIE_TOP - tree_lowest),
                    TRIE_TOP - tree_lowest, TRIE_TOP - lowest))
    return lines


def new_axes() -> "tuple":
    """Axes that fill the whole figure.

    The default subplot rect wastes about a fifth of each dimension on margins,
    which matters on a figure this dense: filling the canvas buys roughly 29%
    more linear resolution for free.
    """
    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=RENDER_DPI)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0, FIGURE_WIDTH)
    axis.set_ylim(0, FIGURE_HEIGHT)
    axis.set_aspect("equal")
    axis.axis("off")
    return (figure, axis)


def draw_node(axis: "plt.Axes", position: "tuple[float, float]", node: int,
              color: str, alpha: float) -> None:
    """Light gray trie node, or a dark numbered leaf."""
    if alpha <= 0.01:
        return
    is_leaf = node in LEAF_LABEL
    if is_leaf:
        radius = LEAF_RADIUS
        fill = LEAF_FILL
    else:
        radius = NODE_RADIUS
        fill = NODE_FILL
    if color != "":
        fill = color
    shadow = patches.Circle((position[0] + 0.025, position[1] - 0.025), radius,
                            facecolor=SHADOW, edgecolor="none", zorder=4,
                            alpha=alpha * 0.9)
    axis.add_patch(shadow)
    circle = patches.Circle(position, radius, facecolor=fill, edgecolor="none",
                            zorder=5, alpha=alpha)
    axis.add_patch(circle)
    if is_leaf:
        axis.text(position[0], position[1], str(LEAF_LABEL[node]), fontsize=LEAF_SIZE,
                  color="white", ha="center", va="center", fontproperties=OPTIMA,
                  zorder=6, alpha=alpha)
    if node == 0:
        axis.text(position[0], position[1], "root", fontsize=8.5, color=TEXT_DARK,
                  ha="center", va="center", fontproperties=OPTIMA_ITALIC, zorder=6,
                  alpha=alpha)


def draw_edge(axis: "plt.Axes", start: "tuple[float, float]",
              end: "tuple[float, float]", color: str, width: float,
              alpha: float, arrow: bool, trim_start: float,
              trim_end: float) -> None:
    """Edge trimmed by whatever part of each endpoint node is still visible."""
    if alpha <= 0.01:
        return
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length < 1e-6:
        return
    unit_x = (end[0] - start[0]) / length
    unit_y = (end[1] - start[1]) / length
    if length <= trim_start + trim_end:
        trim_start = 0.0
        trim_end = 0.0
    a = (start[0] + unit_x * trim_start, start[1] + unit_y * trim_start)
    b = (end[0] - unit_x * trim_end, end[1] - unit_y * trim_end)
    if arrow:
        style = "-|>"
    else:
        style = "-"
    # shrinkA/shrinkB default to 2 points each, which would clip a few pixels off
    # every segment and break a collapsed chain into dashes. Trimming is done above.
    patch = patches.FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8,
                                    linewidth=width, color=color, zorder=3,
                                    alpha=alpha, joinstyle="round",
                                    capstyle="round", shrinkA=0.0, shrinkB=0.0)
    axis.add_patch(patch)


def node_trim(node: int, visibility: float) -> float:
    """How far an edge should stop short of a node, given how visible it is."""
    if node in LEAF_LABEL:
        radius = LEAF_RADIUS
    else:
        radius = NODE_RADIUS
    return (radius + EDGE_GAP) * visibility


def draw_letter(axis: "plt.Axes", position: "tuple[float, float]", character: str,
                color: str, alpha: float) -> None:
    if alpha <= 0.01:
        return
    axis.text(position[0], position[1], character, fontsize=LETTER_SIZE, color=color,
              ha="center", va="center", fontproperties=COURIER, zorder=7, alpha=alpha)


def draw_readout(axis: "plt.Axes", readout: "list[tuple[str, str]]") -> None:
    """One Courier line at the bottom, coloured piecewise."""
    if len(readout) == 0:
        return
    total = 0
    for piece, color in readout:
        total = total + len(piece)
    width_per_char = 0.148
    x = FIGURE_WIDTH / 2 - total * width_per_char / 2
    for piece, color in readout:
        axis.text(x, READOUT_Y, piece, fontsize=READOUT_SIZE, color=color,
                  ha="left", va="center", fontproperties=COURIER, zorder=8)
        x = x + len(piece) * width_per_char


INTERMEDIATE_TARGET = COLLAPSED_XY


def ease(fraction: float) -> float:
    """Cosine ease so the collapse starts and stops gently."""
    clamped = min(max(fraction, 0.0), 1.0)
    return 0.5 * (1 - math.cos(math.pi * clamped))


def base_frame(duration: int) -> dict:
    return {
        "revealed": None,
        "node_colors": {},
        "edge_colors": {},
        "readout": [],
        "t": 0.0,
        "duration_ms": duration,
    }


def suffix_readout(step: dict, threaded_shown: int, created_shown: int) -> "list":
    """The suffix at the bottom, blue for what already existed, green for new."""
    suffix = step["suffix"]
    shared = len(step["threaded"])
    pieces = [("%2d  " % step["start"], TEXT_DARK)]
    if threaded_shown > 0:
        pieces.append((suffix[0:shared], BLUE))
        pieces.append((suffix[shared:shared + created_shown], GREEN))
        remaining = suffix[shared + created_shown:]
    else:
        pieces.append((suffix[0:created_shown], GREEN))
        remaining = suffix[created_shown:]
    if len(remaining) > 0:
        pieces.append((remaining, FADED_GRAY))
    return pieces


def build_trie_specs() -> "list[dict]":
    """The trie growing one suffix at a time."""
    specs = []
    revealed = set([0])

    opening = base_frame(TRIE_FIRST_MS)
    opening["revealed"] = set(revealed)
    opening["readout"] = [(TEXT, TEXT_DARK)]
    specs.append(opening)

    for step in PLAN:
        shared = len(step["threaded"])
        if shared > 0:
            thread = base_frame(TRIE_THREAD_MS)
            thread["revealed"] = set(revealed)
            for node in step["threaded"]:
                thread["node_colors"][node] = BLUE
                thread["edge_colors"][node] = BLUE
            thread["readout"] = suffix_readout(step, shared, 0)
            specs.append(thread)

        created_shown = 0
        for node in step["created"]:
            revealed.add(node)
            created_shown = created_shown + 1
            if node == step["leaf"]:
                duration = TRIE_LEAF_MS
            else:
                duration = TRIE_LETTER_MS
            frame = base_frame(duration)
            frame["revealed"] = set(revealed)
            for earlier in step["threaded"]:
                frame["node_colors"][earlier] = BLUE
                frame["edge_colors"][earlier] = BLUE
            index = 0
            while index < created_shown:
                fresh = step["created"][index]
                if fresh != step["leaf"]:
                    frame["node_colors"][fresh] = GREEN
                frame["edge_colors"][fresh] = GREEN
                index = index + 1
            frame["readout"] = suffix_readout(step, shared, created_shown)
            specs.append(frame)

        settle = base_frame(TRIE_SETTLE_MS)
        settle["revealed"] = set(revealed)
        settle["readout"] = suffix_readout(step, shared, len(step["created"]))
        specs.append(settle)

    final = base_frame(FINAL_HOLD_MS)
    final["revealed"] = set(revealed)
    final["readout"] = [(TEXT, TEXT_DARK)]
    specs.append(final)
    return specs


def collapsing_nodes() -> set:
    """The nodes that disappear when the trie becomes the tree."""
    result = set()
    for node in INCOMING:
        if node not in SURVIVORS:
            result.add(node)
    return result


COLLAPSING = collapsing_nodes()


def build_compress_specs() -> "list[dict]":
    """The trie collapsing into the tree: dots vanish, letters slide together."""
    specs = []

    intro = base_frame(COMPRESS_INTRO_MS)
    intro["revealed"] = set(CHILDREN.keys())
    specs.append(intro)

    mark = base_frame(COMPRESS_MARK_MS)
    mark["revealed"] = set(CHILDREN.keys())
    for node in COLLAPSING:
        mark["node_colors"][node] = ORANGE
    specs.append(mark)

    frame_index = 1
    while frame_index <= COMPRESS_MOTION_FRAMES:
        fraction = ease(frame_index / COMPRESS_MOTION_FRAMES)
        frame = base_frame(COMPRESS_MOTION_MS)
        frame["revealed"] = set(CHILDREN.keys())
        frame["t"] = fraction
        specs.append(frame)
        frame_index = frame_index + 1

    final = base_frame(FINAL_HOLD_MS)
    final["revealed"] = set(CHILDREN.keys())
    final["t"] = 1.0
    specs.append(final)
    return specs


def blend(start: "tuple[float, float]", end: "tuple[float, float]",
          fraction: float) -> "tuple[float, float]":
    return (start[0] + (end[0] - start[0]) * fraction,
            start[1] + (end[1] - start[1]) * fraction)


def frame_geometry(spec: dict) -> "tuple[dict, dict, dict]":
    """Node positions, letter positions, and node alphas for one frame."""
    fraction = spec["t"]
    node_xy = {}
    letter_xy = {}
    alpha = {}
    for node in CHILDREN:
        if fraction <= 0.0:
            node_xy[node] = TRIE_XY[node]
            alpha[node] = 1.0
        elif node in SURVIVORS:
            node_xy[node] = blend(TRIE_XY[node], TREE_XY[node], fraction)
            alpha[node] = 1.0
        else:
            node_xy[node] = blend(TRIE_XY[node], INTERMEDIATE_TARGET[node], fraction)
            alpha[node] = max(0.0, 1.0 - fraction * 1.6)
    for node in INCOMING:
        if fraction <= 0.0:
            letter_xy[node] = LETTER_TRIE[node]
        else:
            letter_xy[node] = blend(LETTER_TRIE[node], LETTER_TREE[node], fraction)
    return (node_xy, letter_xy, alpha)


def draw_frame(spec: dict, output_path: str) -> None:
    """Render one frame of either animation."""
    figure, axis = new_axes()
    node_xy, letter_xy, alpha = frame_geometry(spec)
    revealed = spec["revealed"]
    fraction = spec["t"]

    for node in INCOMING:
        if node not in revealed:
            continue
        parent = PARENT[node]
        color = spec["edge_colors"].get(node, EDGE_GRAY)
        if color == EDGE_GRAY:
            width = 0.9
        else:
            width = 1.9
        if fraction > 0.08 and node not in SURVIVORS:
            arrow = False
        else:
            arrow = True
        draw_edge(axis, node_xy[parent], node_xy[node], color, width, 1.0, arrow,
                  node_trim(parent, alpha[parent]), node_trim(node, alpha[node]))

    for node in INCOMING:
        if node not in revealed:
            continue
        color = spec["edge_colors"].get(node, TEXT_DARK)
        draw_letter(axis, letter_xy[node], INCOMING[node], color, 1.0)

    for node in CHILDREN:
        if node not in revealed:
            continue
        draw_node(axis, node_xy[node], node, spec["node_colors"].get(node, ""),
                  alpha[node])

    draw_readout(axis, spec["readout"])
    figure.savefig(output_path, facecolor="white")
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: suffix_tree.py trie|compress OUTPUT.gif")
        return
    which = sys.argv[1]
    output_path = sys.argv[2]

    print("Structural checks:")
    for line in verify():
        print("  ok: " + line)

    if which == "trie":
        specs = build_trie_specs()
    elif which == "compress":
        specs = build_compress_specs()
    else:
        print("first argument must be trie or compress")
        return

    durations = []
    for spec in specs:
        durations.append(spec["duration_ms"])
    print("Rendering %d frames at %dx%d, %.1f s of playback..."
          % (len(specs), OUTPUT_WIDTH, OUTPUT_HEIGHT, sum(durations) / 1000.0))
    directory = tempfile.mkdtemp(prefix="suffix_" + which + "_")
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
