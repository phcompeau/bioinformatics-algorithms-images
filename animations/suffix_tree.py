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

Run:  python3 example_suffix_tree.py trie OUTPUT.gif
      python3 example_suffix_tree.py compress OUTPUT.gif
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

from lecture_style import mono_advance
from make_gif import assemble_transparent_gif

# Styled after Phillip's 02-180 Lecture 5 (Read_Mapping.pptx, slides 141-155 for
# the trie and 204-233 for the compression): cream ground, white nodes with a
# thin dark outline, large monospace letters, no arrowheads.
BACKGROUND = "#EEE9DF"
INK = "#1A1A1A"
NODE_FILL = "#FFFFFF"
LEAF_FILL = "#3F3F3F"
BLUE = "#176FC1"
GREEN = "#149B52"
RED = "#ED1C24"
FADED = "#C9C4B8"
TEXT_DARK = "#262626"

# His deck sets the letters in Consolas, which is not installed here; Menlo is
# the closest metric and tone match on macOS.
MONO = font_manager.FontProperties(family="Menlo")
OPTIMA = font_manager.FontProperties(family="Optima")
OPTIMA_ITALIC = font_manager.FontProperties(family="Optima", style="italic")

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
FIGURE_HEIGHT = 10.0
RENDER_DPI = 120
OUTPUT_WIDTH = 1320
OUTPUT_HEIGHT = 1200

# Measured from the lecture deck: nodes 0.25 in across, root 0.417 in, letters
# 18 pt, leaf indices 14 pt white on a dark disc.
NODE_RADIUS = 0.125
LEAF_RADIUS = 0.150
ROOT_RADIUS = 0.208
LETTER_SIZE = 17.0
LEAF_SIZE = 11.5
ROOT_SIZE = 10.0
EDGE_WIDTH = 0.9
EDGE_WIDTH_MARKED = 2.2

TRIE_TOP = 8.60
TRIE_GAP = 0.56
LEFT_MARGIN = 0.45
EDGE_GAP = 0.025
LETTER_OFFSET = 0.19
# A letter sits this far back from its child node along its own edge, rather than
# at the edge midpoint. On the root's long shallow edges that keeps each label
# beside the child it belongs to, which is how the lecture slides place them.
LETTER_BACKOFF = 0.62
LETTER_BACKOFF_FRACTION = 0.45

# One character of 17 pt Menlo advances about 0.14 in; a collapsed edge is made
# long enough to seat its whole label plus a little breathing room.
CHAR_ADVANCE = 0.153
LABEL_END_GAP = 0.28
LABEL_END_PAD = 0.32
# The root fans out across the whole width, so its edges need a real vertical
# drop or they come out nearly parallel and their labels pile up.
MIN_DROP = 1.25
TREE_LABEL_OFFSET = 0.150

# The text rides at the top of the frame, large, with the suffix being inserted
# picked out inside it: the viewer sees where in the text the suffix comes from
# instead of reading it twice.
HEADER_Y = 9.52
HEADER_SIZE = 36.0
HEADER_ADVANCE = HEADER_SIZE / 72.0 * 0.60

# Clearance budget for label placement. A 17 pt Menlo glyph is about 0.21 in
# across, so treat each letter as a disc of this radius and keep it off the
# nodes, off other edges, and off its neighbours.
LETTER_DISC = 0.105
NODE_CLEAR = 0.055
EDGE_CLEAR = 0.070
LETTER_CLEAR = 0.190

TRIE_THREAD_MS = 900
TRIE_LETTER_MS = 220
TRIE_LEAF_MS = 800
TRIE_SETTLE_MS = 360
TRIE_FIRST_MS = 1400
FINAL_HOLD_MS = 4600

COMPRESS_INTRO_MS = 3200
COMPRESS_MARK_MS = 3000
COMPRESS_MOTION_FRAMES = 28
COMPRESS_MOTION_MS = 120


def child_sort_key(character: str) -> str:
    """The book orders root branches $ then alphabetically."""
    if character == "$":
        return " "
    return character


def build_trie() -> "tuple[dict, dict, dict]":
    """children[node][char] = child, leaf_label[node] = suffix start, incoming char."""
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
    result = {}
    for node in DEPTH:
        result[node] = (X[node], TRIE_TOP - DEPTH[node] * TRIE_GAP)
    return result


TRIE_XY = trie_positions()


def build_tree() -> "tuple[dict, dict]":
    """Collapse non-branching paths.

    Returns edges[top] = {label: bottom} and owner[node] = (top, bottom, index),
    saying which collapsed edge each trie node's letter belongs to.
    """
    edges = {}
    owner = {}
    pending = [0]
    while len(pending) > 0:
        top = pending.pop()
        edges[top] = {}
        for child in sorted_children(top):
            label = INCOMING[child]
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


def label_of(top: int, bottom: int) -> str:
    """The collapsed label on the edge between two surviving nodes."""
    for label in TREE_EDGES[top]:
        if TREE_EDGES[top][label] == bottom:
            return label
    raise ValueError("no collapsed edge there")


def tree_positions() -> dict:
    """Surviving node centers after the collapse.

    x never changes. Each edge drops far enough that its rotated label fits
    along it, which is what makes the figure shrink vertically.
    """
    result = {0: (X[0], TRIE_TOP)}
    pending = [0]
    while len(pending) > 0:
        top = pending.pop()
        for label in TREE_EDGES[top]:
            bottom = TREE_EDGES[top][label]
            needed = len(label) * CHAR_ADVANCE + LABEL_END_GAP + LABEL_END_PAD
            dx = X[bottom] - X[top]
            squared = needed * needed - dx * dx
            if squared < MIN_DROP * MIN_DROP:
                drop = MIN_DROP
            else:
                drop = math.sqrt(squared)
            result[bottom] = (X[bottom], result[top][1] - drop)
            if bottom in TREE_EDGES:
                pending.append(bottom)
    return result


TREE_XY = tree_positions()
SURVIVORS = set(TREE_XY.keys())


def node_radius(node: int) -> float:
    if node == 0:
        return ROOT_RADIUS
    if node in LEAF_LABEL:
        return LEAF_RADIUS
    return NODE_RADIUS


def point_segment_distance(px: float, py: float, ax: float, ay: float,
                           bx: float, by: float) -> float:
    """Distance from a point to a segment, not to the infinite line."""
    dx = bx - ax
    dy = by - ay
    squared = dx * dx + dy * dy
    if squared < 1e-12:
        return math.hypot(px - ax, py - ay)
    along = ((px - ax) * dx + (py - ay) * dy) / squared
    along = min(max(along, 0.0), 1.0)
    return math.hypot(px - (ax + along * dx), py - (ay + along * dy))


def trie_segments(positions: dict) -> "list[tuple]":
    """(child, ax, ay, bx, by) for every trie edge, keyed by the child it feeds."""
    result = []
    for node in INCOMING:
        ax, ay = positions[PARENT[node]]
        bx, by = positions[node]
        result.append((node, ax, ay, bx, by))
    return result


def placement_margin(x: float, y: float, own_edges: set, positions: dict,
                     segments: "list[tuple]", placed: "list[tuple]") -> float:
    """How much room a glyph at (x, y) has. Negative means it is colliding.

    Measured against every node disc, every edge it does not belong to, and
    every letter already placed. This is what keeps a letter such as the `b` on
    the root's `ab...` path from landing on top of a node.
    """
    margin = 99.0
    for node in positions:
        gap = (math.hypot(x - positions[node][0], y - positions[node][1])
               - (node_radius(node) + LETTER_DISC + NODE_CLEAR))
        if gap < margin:
            margin = gap
    for child, ax, ay, bx, by in segments:
        if child in own_edges:
            continue
        gap = (point_segment_distance(x, y, ax, ay, bx, by)
               - (LETTER_DISC + EDGE_CLEAR))
        if gap < margin:
            margin = gap
    for other_x, other_y in placed:
        gap = math.hypot(x - other_x, y - other_y) - LETTER_CLEAR
        if gap < margin:
            margin = gap
    return margin


def trie_letter_candidates(node: int, positions: dict) -> "list[tuple]":
    """Placements for one trie letter, nearest to the house default first."""
    ax, ay = positions[PARENT[node]]
    bx, by = positions[node]
    length = math.hypot(bx - ax, by - ay)
    unit_x = (bx - ax) / length
    unit_y = (by - ay) / length
    default_backoff = min(LETTER_BACKOFF, length * LETTER_BACKOFF_FRACTION)
    default_x = bx - unit_x * default_backoff - unit_y * LETTER_OFFSET
    default_y = by - unit_y * default_backoff + unit_x * LETTER_OFFSET
    scored = []
    for sign in (1.0, -1.0):
        for offset in (0.19, 0.24, 0.30, 0.37, 0.45):
            for fraction in (0.45, 0.32, 0.60, 0.22, 0.78):
                backoff = min(LETTER_BACKOFF, length * fraction)
                x = bx - unit_x * backoff - sign * unit_y * offset
                y = by - unit_y * backoff + sign * unit_x * offset
                shift = math.hypot(x - default_x, y - default_y)
                scored.append((shift, x, y))
    scored.sort()
    result = []
    for shift, x, y in scored:
        result.append((x, y))
    return result


def letters_beside(positions: dict) -> dict:
    """Trie letters: upright, beside their own edge, nudged clear of collisions.

    Every letter starts at the house default (a fixed back-off from its child,
    offset to one side). Letters are then placed shallowest first, and one is
    moved only when its default would collide, and then by the smallest amount
    that clears. The figure keeps its regular look and stops overlapping.
    """
    segments = trie_segments(positions)
    ordering = []
    for node in INCOMING:
        ordering.append((DEPTH[node], positions[node][0], node))
    ordering.sort()
    result = {}
    placed = []
    for depth, x_of_node, node in ordering:
        own_edges = set([node])
        for child in sorted_children(node):
            own_edges.add(child)
        best_x = 0.0
        best_y = 0.0
        best_margin = -99.0
        for candidate_x, candidate_y in trie_letter_candidates(node, positions):
            margin = placement_margin(candidate_x, candidate_y, own_edges,
                                      positions, segments, placed)
            if margin > best_margin:
                best_margin = margin
                best_x = candidate_x
                best_y = candidate_y
            if margin >= 0.0:
                break
        result[node] = (best_x, best_y, 0.0)
        placed.append((best_x, best_y))
    return result


def tree_edge_list() -> "list[tuple]":
    """(top, bottom, label), longest label first: the tightest ones place first."""
    result = []
    for top in TREE_EDGES:
        for label in TREE_EDGES[top]:
            result.append((-len(label), top, TREE_EDGES[top][label], label))
    result.sort()
    ordered = []
    for negative_length, top, bottom, label in result:
        ordered.append((top, bottom, label))
    return ordered


def tree_segments() -> "list[tuple]":
    """(bottom, ax, ay, bx, by) for every collapsed edge, keyed by its lower end."""
    result = []
    for top in TREE_EDGES:
        for label in TREE_EDGES[top]:
            bottom = TREE_EDGES[top][label]
            result.append((bottom, TREE_XY[top][0], TREE_XY[top][1],
                           TREE_XY[bottom][0], TREE_XY[bottom][1]))
    return result


def run_placements(top: int, bottom: int, label: str, offset: float,
                   slide: float) -> "list[tuple]":
    """Where each character of one collapsed label lands, given an offset.

    `offset` is signed distance to the side of the edge, `slide` moves the whole
    run along the edge. The rotation is normalized into [-90, 90] so a label
    never renders upside down, exactly as the lecture slides place them.
    """
    ax, ay = TREE_XY[top]
    bx, by = TREE_XY[bottom]
    length = math.hypot(bx - ax, by - ay)
    angle = math.degrees(math.atan2(by - ay, bx - ax))
    if angle > 90.0:
        angle = angle - 180.0
    elif angle < -90.0:
        angle = angle + 180.0
    radians = math.radians(angle)
    unit_x = math.cos(radians)
    unit_y = math.sin(radians)
    toward_x = (bx - ax) / length
    toward_y = (by - ay) / length
    # Anchor the run just short of the child, so labels on nearly parallel edges
    # separate along with the children they belong to.
    centre_distance = LABEL_END_GAP + len(label) * CHAR_ADVANCE / 2.0 + slide
    centre_x = bx - toward_x * centre_distance
    centre_y = by - toward_y * centre_distance
    result = []
    index = 0
    while index < len(label):
        step = index - (len(label) - 1) / 2.0
        result.append((centre_x + unit_x * step * CHAR_ADVANCE - unit_y * offset,
                       centre_y + unit_y * step * CHAR_ADVANCE + unit_x * offset,
                       angle))
        index = index + 1
    return result


def run_margin(placements: "list[tuple]", own: set, segments: "list[tuple]",
               placed: "list[tuple]") -> float:
    """The tightest clearance any character of one label has."""
    margin = 99.0
    for x, y, angle in placements:
        local = placement_margin(x, y, own, TREE_XY, segments, placed)
        if local < margin:
            margin = local
    return margin


def letters_along_edges() -> dict:
    """Tree letters: set along their collapsed edge, rotated to follow it.

    Each label is placed as a whole run, longest first, and slid or flipped to
    the other side of its edge only when the default collides. This is what
    keeps the root's `panamabananas$` off its neighbouring `s$`: those two edges
    end up nearly parallel, so the long label has to take the far side.
    """
    segments = tree_segments()
    placed = []
    chosen = {}
    for top, bottom, label in tree_edge_list():
        own = set([bottom])
        if bottom in TREE_EDGES:
            for other_label in TREE_EDGES[bottom]:
                own.add(TREE_EDGES[bottom][other_label])
        best = None
        best_margin = -99.0
        for offset in (TREE_LABEL_OFFSET, -TREE_LABEL_OFFSET, 0.19, -0.19,
                       0.24, -0.24, 0.30, -0.30, 0.38, -0.38, 0.47, -0.47):
            for slide in (0.0, 0.16, -0.16, 0.34, -0.34, 0.55, 0.80):
                placements = run_placements(top, bottom, label, offset, slide)
                margin = run_margin(placements, own, segments, placed)
                if margin > best_margin:
                    best_margin = margin
                    best = placements
                if margin >= 0.0:
                    break
            if best_margin >= 0.0:
                break
        chosen[(top, bottom)] = best
        for x, y, angle in best:
            placed.append((x, y))

    result = {}
    for node in INCOMING:
        if node not in OWNER:
            continue
        top, bottom, index = OWNER[node]
        result[node] = chosen[(top, bottom)][index]
    return result


def collapsed_positions() -> dict:
    """Every node's position after the collapse; discarded ones ride their edge."""
    result = {}
    for node in TREE_XY:
        result[node] = TREE_XY[node]
    for node in INCOMING:
        if node in SURVIVORS:
            continue
        top, bottom, index = OWNER[node]
        label = label_of(top, bottom)
        fraction = (index + 1.0) / len(label)
        ax, ay = TREE_XY[top]
        bx, by = TREE_XY[bottom]
        result[node] = (ax + (bx - ax) * fraction, ay + (by - ay) * fraction)
    return result


COLLAPSED_XY = collapsed_positions()
LETTER_TRIE = letters_beside(TRIE_XY)
LETTER_TREE = letters_along_edges()


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
        plan.append({"start": start, "suffix": suffix, "threaded": threaded,
                     "created": created, "leaf": path[len(path) - 1]})
        start = start + 1
    return plan


PLAN = insertion_plan()


def verify() -> "list[str]":
    """Assert both structures against the published figures and the layout."""
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
    assert "".join(branches) == PUBLISHED_ROOT_BRANCHES, "root branches disagree"
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
            assert spell(top) + label == spell(TREE_EDGES[top][label]), (
                "collapsed label %s does not spell the gap it covers" % label)
    lines.append("every collapsed label spells the gap between its endpoints")

    for node in LEAF_LABEL:
        assert node in SURVIVORS, "a leaf was collapsed away"
    lines.append("all %d leaves and %d branch nodes survive the collapse"
                 % (len(LEAF_LABEL), len(SURVIVORS) - len(LEAF_LABEL)))

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

    # Every collapsed edge must be long enough to seat its own rotated label.
    for top in TREE_EDGES:
        for label in TREE_EDGES[top]:
            bottom = TREE_EDGES[top][label]
            ax, ay = TREE_XY[top]
            bx, by = TREE_XY[bottom]
            length = math.hypot(bx - ax, by - ay)
            needed = len(label) * CHAR_ADVANCE + LABEL_END_GAP
            assert length >= needed, (
                "edge %r is %.2f in but its label needs %.2f in" % (label, length, needed))
    lines.append("every collapsed edge is long enough to seat its rotated label")

    # Every letter in both figures must clear every node, every edge it does not
    # belong to, and every other letter. This is the check that catches the `b`
    # on the root's `ab...` path sitting on its node.
    trie_edges = trie_segments(TRIE_XY)
    tightest = 99.0
    offender = ""
    for node in LETTER_TRIE:
        own = set([node])
        for child in sorted_children(node):
            own.add(child)
        others = []
        for other in LETTER_TRIE:
            if other != node:
                others.append((LETTER_TRIE[other][0], LETTER_TRIE[other][1]))
        margin = placement_margin(LETTER_TRIE[node][0], LETTER_TRIE[node][1], own,
                                  TRIE_XY, trie_edges, others)
        if margin < tightest:
            tightest = margin
            offender = "%r at depth %d" % (INCOMING[node], DEPTH[node])
    assert tightest >= 0.0, ("trie letter %s overlaps something by %.3f in"
                             % (offender, -tightest))
    lines.append("every trie letter clears its neighbours, the nodes and the edges "
                 "(tightest %.3f in of slack, %s)" % (tightest, offender))

    collapsed_edges = tree_segments()
    tightest_tree = 99.0
    offender_tree = ""
    for node in LETTER_TREE:
        top, bottom, index = OWNER[node]
        own = set([bottom])
        if bottom in TREE_EDGES:
            for label in TREE_EDGES[bottom]:
                own.add(TREE_EDGES[bottom][label])
        others = []
        for other in LETTER_TREE:
            if OWNER[other][1] != bottom:
                others.append((LETTER_TREE[other][0], LETTER_TREE[other][1]))
        margin = placement_margin(LETTER_TREE[node][0], LETTER_TREE[node][1], own,
                                  TREE_XY, collapsed_edges, others)
        if margin < tightest_tree:
            tightest_tree = margin
            offender_tree = "%r in label %r" % (INCOMING[node], label_of(top, bottom))
    assert tightest_tree >= 0.0, ("suffix tree letter %s overlaps something by %.3f in"
                                  % (offender_tree, -tightest_tree))
    lines.append("every suffix tree label clears the other labels and the nodes "
                 "(tightest %.3f in of slack, %s)" % (tightest_tree, offender_tree))

    lowest = TRIE_TOP
    for node in TRIE_XY:
        if TRIE_XY[node][1] < lowest:
            lowest = TRIE_XY[node][1]
    assert lowest - LEAF_RADIUS > 0.25, "the trie runs off the bottom"
    assert TRIE_TOP + ROOT_RADIUS < HEADER_Y - 0.45, "the trie collides with the text"
    lines.append("trie spans %.2f to %.2f in, clear of the text at %.2f in"
                 % (lowest, TRIE_TOP, HEADER_Y))
    header_width = len(TEXT) * HEADER_ADVANCE
    assert header_width < FIGURE_WIDTH - 0.6, "the text is too wide for the frame"
    lines.append("the text reads %.1f in wide at %.0f pt across the top"
                 % (header_width, HEADER_SIZE))

    tree_lowest = TRIE_TOP
    for node in TREE_XY:
        if TREE_XY[node][1] < tree_lowest:
            tree_lowest = TREE_XY[node][1]
    lines.append("the tree is %.1fx shorter than the trie (%.2f vs %.2f in tall)"
                 % ((TRIE_TOP - lowest) / (TRIE_TOP - tree_lowest),
                    TRIE_TOP - tree_lowest, TRIE_TOP - lowest))
    return lines


def new_axes() -> "tuple":
    """Cream axes filling the whole figure."""
    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=RENDER_DPI)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0, FIGURE_WIDTH)
    axis.set_ylim(0, FIGURE_HEIGHT)
    axis.set_aspect("equal")
    axis.axis("off")
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    return (figure, axis)


def draw_node(axis: "plt.Axes", position: "tuple[float, float]", node: int,
              outline: str, alpha: float) -> None:
    """White node with a thin dark outline; leaves are dark discs with an index."""
    if alpha <= 0.01:
        return
    radius = node_radius(node)
    if node in LEAF_LABEL:
        face = LEAF_FILL
        edge = LEAF_FILL
        width = 1.0
    else:
        face = NODE_FILL
        edge = INK
        width = 1.1
    if outline != "":
        edge = outline
        width = 2.2
        if node not in LEAF_LABEL:
            face = NODE_FILL
    circle = patches.Circle(position, radius, facecolor=face, edgecolor=edge,
                            linewidth=width, zorder=5, alpha=alpha)
    axis.add_patch(circle)
    if node in LEAF_LABEL:
        axis.text(position[0], position[1], str(LEAF_LABEL[node]), fontsize=LEAF_SIZE,
                  color="white", ha="center", va="center", fontproperties=MONO,
                  zorder=6, alpha=alpha)
    if node == 0:
        axis.text(position[0], position[1], "Root", fontsize=ROOT_SIZE, color=INK,
                  ha="center", va="center", fontproperties=OPTIMA, zorder=6,
                  alpha=alpha)


def draw_edge(axis: "plt.Axes", start: "tuple[float, float]",
              end: "tuple[float, float]", color: str, width: float,
              trim_start: float, trim_end: float) -> None:
    """Plain line, no arrowhead, trimmed by the visible part of each node."""
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length < 1e-6:
        return
    unit_x = (end[0] - start[0]) / length
    unit_y = (end[1] - start[1]) / length
    if length <= trim_start + trim_end:
        trim_start = 0.0
        trim_end = 0.0
    axis.plot([start[0] + unit_x * trim_start, end[0] - unit_x * trim_end],
              [start[1] + unit_y * trim_start, end[1] - unit_y * trim_end],
              color=color, linewidth=width, solid_capstyle="round", zorder=3)


def draw_letter(axis: "plt.Axes", placement: "tuple[float, float, float]",
                character: str, color: str) -> None:
    axis.text(placement[0], placement[1], character, fontsize=LETTER_SIZE,
              color=color, ha="center", va="center", fontproperties=MONO,
              rotation=placement[2], rotation_mode="anchor", zorder=7)


def draw_header(axis: "plt.Axes", figure: "plt.Figure", readout: "list") -> None:
    """The text across the top, large, coloured piece by piece."""
    if len(readout) == 0:
        return
    advance = mono_advance(figure, HEADER_SIZE)
    total = 0
    for piece, color in readout:
        total = total + len(piece)
    x = FIGURE_WIDTH / 2 - total * advance / 2
    for piece, color in readout:
        axis.text(x, HEADER_Y, piece, fontsize=HEADER_SIZE, color=color,
                  ha="left", va="center", fontproperties=MONO, zorder=8)
        x = x + len(piece) * advance


def ease(fraction: float) -> float:
    clamped = min(max(fraction, 0.0), 1.0)
    return 0.5 * (1 - math.cos(math.pi * clamped))


def base_frame(duration: int) -> dict:
    return {"revealed": None, "node_outlines": {}, "letter_colors": {},
            "edge_colors": {}, "readout": [], "t": 0.0, "duration_ms": duration}


def suffix_readout(step: dict, created_shown: int) -> "list":
    """The whole text with the suffix under insertion picked out inside it.

    Everything before the suffix stays faint, the part of the suffix that already
    had a path is blue, the part being added right now is green, and the rest is
    faint until it is reached. The viewer never has to match up two copies of the
    string, because there is only ever one.
    """
    suffix = step["suffix"]
    shared = len(step["threaded"])
    pieces = []
    if step["start"] > 0:
        pieces.append((TEXT[0:step["start"]], FADED))
    if shared > 0:
        pieces.append((suffix[0:shared], BLUE))
    if created_shown > 0:
        pieces.append((suffix[shared:shared + created_shown], GREEN))
    remaining = suffix[shared + created_shown:]
    if len(remaining) > 0:
        pieces.append((remaining, FADED))
    return pieces


def build_trie_specs() -> "list[dict]":
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
                thread["node_outlines"][node] = BLUE
                thread["edge_colors"][node] = BLUE
                thread["letter_colors"][node] = BLUE
            thread["readout"] = suffix_readout(step, 0)
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
                frame["node_outlines"][earlier] = BLUE
                frame["edge_colors"][earlier] = BLUE
                frame["letter_colors"][earlier] = BLUE
            index = 0
            while index < created_shown:
                fresh = step["created"][index]
                if fresh != step["leaf"]:
                    frame["node_outlines"][fresh] = GREEN
                frame["edge_colors"][fresh] = GREEN
                frame["letter_colors"][fresh] = GREEN
                index = index + 1
            frame["readout"] = suffix_readout(step, created_shown)
            specs.append(frame)

        settle = base_frame(TRIE_SETTLE_MS)
        settle["revealed"] = set(revealed)
        settle["readout"] = suffix_readout(step, len(step["created"]))
        specs.append(settle)

    final = base_frame(FINAL_HOLD_MS)
    final["revealed"] = set(revealed)
    final["readout"] = [(TEXT, TEXT_DARK)]
    specs.append(final)
    return specs


def collapsing_nodes() -> set:
    result = set()
    for node in INCOMING:
        if node not in SURVIVORS:
            result.add(node)
    return result


COLLAPSING = collapsing_nodes()


def build_compress_specs() -> "list[dict]":
    specs = []
    everything = set(CHILDREN.keys())

    header = [(TEXT, TEXT_DARK)]

    intro = base_frame(COMPRESS_INTRO_MS)
    intro["revealed"] = set(everything)
    intro["readout"] = header
    specs.append(intro)

    mark = base_frame(COMPRESS_MARK_MS)
    mark["revealed"] = set(everything)
    mark["readout"] = header
    for node in COLLAPSING:
        mark["node_outlines"][node] = RED
    specs.append(mark)

    frame_index = 1
    while frame_index <= COMPRESS_MOTION_FRAMES:
        frame = base_frame(COMPRESS_MOTION_MS)
        frame["revealed"] = set(everything)
        frame["readout"] = header
        frame["t"] = ease(frame_index / COMPRESS_MOTION_FRAMES)
        specs.append(frame)
        frame_index = frame_index + 1

    final = base_frame(FINAL_HOLD_MS)
    final["revealed"] = set(everything)
    final["readout"] = header
    final["t"] = 1.0
    specs.append(final)
    return specs


def blend(start: float, end: float, fraction: float) -> float:
    return start + (end - start) * fraction


def frame_geometry(spec: dict) -> "tuple[dict, dict, dict]":
    fraction = spec["t"]
    node_xy = {}
    letters = {}
    alpha = {}
    for node in CHILDREN:
        if fraction <= 0.0:
            node_xy[node] = TRIE_XY[node]
            alpha[node] = 1.0
        elif node in SURVIVORS:
            node_xy[node] = (blend(TRIE_XY[node][0], TREE_XY[node][0], fraction),
                             blend(TRIE_XY[node][1], TREE_XY[node][1], fraction))
            alpha[node] = 1.0
        else:
            node_xy[node] = (blend(TRIE_XY[node][0], COLLAPSED_XY[node][0], fraction),
                             blend(TRIE_XY[node][1], COLLAPSED_XY[node][1], fraction))
            alpha[node] = max(0.0, 1.0 - fraction * 1.6)
    for node in INCOMING:
        if fraction <= 0.0:
            letters[node] = LETTER_TRIE[node]
        else:
            a = LETTER_TRIE[node]
            b = LETTER_TREE[node]
            letters[node] = (blend(a[0], b[0], fraction), blend(a[1], b[1], fraction),
                             blend(a[2], b[2], fraction))
    return (node_xy, letters, alpha)


def visible_trim(node: int, visibility: float) -> float:
    return (node_radius(node) + EDGE_GAP) * visibility


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes()
    node_xy, letters, alpha = frame_geometry(spec)
    revealed = spec["revealed"]

    for node in INCOMING:
        if node not in revealed:
            continue
        parent = PARENT[node]
        color = spec["edge_colors"].get(node, INK)
        if color == INK:
            width = EDGE_WIDTH
        else:
            width = EDGE_WIDTH_MARKED
        draw_edge(axis, node_xy[parent], node_xy[node], color, width,
                  visible_trim(parent, alpha[parent]), visible_trim(node, alpha[node]))

    for node in INCOMING:
        if node not in revealed:
            continue
        draw_letter(axis, letters[node], INCOMING[node],
                    spec["letter_colors"].get(node, INK))

    for node in CHILDREN:
        if node not in revealed:
            continue
        draw_node(axis, node_xy[node], node, spec["node_outlines"].get(node, ""),
                  alpha[node])

    draw_header(axis, figure, spec["readout"])
    figure.savefig(output_path, transparent=True)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: example_suffix_tree.py trie|compress OUTPUT.gif")
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

    assemble_transparent_gif(frame_paths, output_path, width=OUTPUT_WIDTH,
                             height=OUTPUT_HEIGHT, frame_durations=durations)
    print("Saved " + output_path)


if __name__ == "__main__":
    main()
