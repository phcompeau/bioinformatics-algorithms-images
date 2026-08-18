"""Animated small parsimony, in the 02-180 lecture style.

This is the deck's own worked example (Evolutionary_Trees.pptx slides 190-198):
eight leaves carrying one column of an alignment, C C A C G G T C, on a balanced
rooted tree. Small parsimony treats every column of an alignment independently, so
one column is the whole problem.

Everything the algorithm computes is shown:

  leaves       score 0 for the letter they carry, infinite for the other three
  each node    the four scores s(v, k) filling in one symbol at a time, each with
               the minimum over each child's four candidate sums written out
  the root     its smallest score is the parsimony score of the column
  backtracking the letter each node takes, chosen by the minimum that produced its
               parent's score, then the edges that carry a mutation

Each node's scores are laid out the way the deck lays them out: a header row
A C G T with the values directly beneath, so the columns line up.

Oracles: all 28 score-vector entries and all 7 ancestral letters are asserted
against the numbers printed on slides 194 and 198, and the score is independently
checked against a brute-force search over all 4^7 assignments to the internal
nodes.

Run:  python3 example_small_parsimony.py OUTPUT.gif
"""

import os
import sys
import tempfile

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, BLUE, DIM, DISC, FAINT, GREEN, INK, MONO,
                           MONO_BOLD, OPTIMA, OPTIMA_ITALIC, RED, mono_advance,
                           new_axes)
from make_gif import assemble_transparent_gif

ALPHABET = "ACGT"
# Read off the picture on slide 191: eight leaves, left to right.
LEAF_LETTERS = ["C", "C", "A", "C", "G", "G", "T", "C"]
CHILDREN = {
    "n0": ("L0", "L1"), "n1": ("L2", "L3"), "n2": ("L4", "L5"), "n3": ("L6", "L7"),
    "m0": ("n0", "n1"), "m1": ("n2", "n3"),
    "root": ("m0", "m1"),
}
POSTORDER = ["n0", "n1", "n2", "n3", "m0", "m1", "root"]
# Every score vector printed on slide 194, and every ancestral letter on slide 198.
PUBLISHED_SCORES = {
    "n0": (2, 0, 2, 2), "n1": (1, 1, 2, 2), "n2": (2, 2, 0, 2), "n3": (2, 1, 2, 1),
    "m0": (2, 1, 3, 3), "m1": (3, 2, 2, 2), "root": (5, 3, 4, 4),
}
PUBLISHED_ASSIGNMENT = {"root": "C", "m0": "C", "m1": "C", "n0": "C", "n1": "C",
                        "n2": "G", "n3": "C"}
PUBLISHED_SCORE = 3

FIGURE_WIDTH = 14.6
FIGURE_HEIGHT = 8.2
RENDER_DPI = 105
OUTPUT_WIDTH = 1533
OUTPUT_HEIGHT = 861

LEAF_Y = 2.75
LEVEL1_Y = 4.00
LEVEL2_Y = 5.40
ROOT_Y = 6.80
LEFT_MARGIN = 1.85
LEAF_GAP = 1.70

LEAF_RADIUS = 0.22
NODE_RADIUS = 0.19
LETTER_SIZE = 16.0
# The deck writes A C G T above the four scores so the columns line up.
VECTOR_SPACING = 0.30
VECTOR_SIZE = 13.0
HEADER_SIZE = 13.0
VECTOR_SIDE = 1.55
VECTOR_HEADER_UP = 0.20
VECTOR_VALUE_DOWN = 0.10
LEAF_HEADER_DOWN = 0.55
LEAF_VALUE_DOWN = 0.83
# The root's vector has to clear the two edges leaving it, so it sits fully above.
ROOT_HEADER_UP = 0.66
ROOT_VALUE_UP = 0.38
# Sampled from the published figures this animation sits beside
# (images/Evolution/Sankoff_step_1..4.png): the root is a green disc, the other
# internal nodes are plain grey, and no value is highlighted in red. The deck uses
# red for minima, but the book does not, and the book is where this GIF lands.
BOOK_GREEN = "#5EA668"
BOOK_GREY = "#B3B3B3"

CAPTION_Y = 7.86
LABEL_SIZE = 17.0
WORK_TOP = 1.62
WORK_STEP = 0.42
WORK_SIZE = 14.0
SCORE_SIZE = 21.0
SCORE_X = 1.10
SCORE_Y = 6.95

OPENING_MS = 3600
LEAF_MS = 3400
FIRST_SYMBOL_MS = 2200
SYMBOL_MS = 1500
PICK_MS = 3600
BACK_MS = 2000
FINAL_HOLD_MS = 5600

INFINITY = float("inf")


def leaf_names() -> "list[str]":
    result = []
    index = 0
    while index < len(LEAF_LETTERS):
        result.append("L%d" % index)
        index = index + 1
    return result


LEAVES = leaf_names()


def letter_of() -> dict:
    result = {}
    index = 0
    while index < len(LEAF_LETTERS):
        result["L%d" % index] = LEAF_LETTERS[index]
        index = index + 1
    return result


LETTER_OF = letter_of()


def tree_edges() -> "list[tuple]":
    result = []
    for parent in POSTORDER:
        for child in CHILDREN[parent]:
            result.append((parent, child))
    return result


EDGES = tree_edges()


def positions() -> dict:
    """Leaves evenly spaced; every parent centred over its two children."""
    result = {}
    index = 0
    while index < len(LEAVES):
        result[LEAVES[index]] = (LEFT_MARGIN + index * LEAF_GAP, LEAF_Y)
        index = index + 1
    heights = {"n0": LEVEL1_Y, "n1": LEVEL1_Y, "n2": LEVEL1_Y, "n3": LEVEL1_Y,
               "m0": LEVEL2_Y, "m1": LEVEL2_Y, "root": ROOT_Y}
    for node in POSTORDER:
        first = CHILDREN[node][0]
        second = CHILDREN[node][1]
        result[node] = ((result[first][0] + result[second][0]) / 2, heights[node])
    return result


POSITION = positions()


def sides() -> dict:
    """Where each node's score vector sits, following the deck's own figure.

    Internal vectors go to the left of their node at the node's own height, the
    root's sits above it, and the leaves carry theirs underneath. Keeping every
    internal vector on the same side is what stops the two middle ones colliding.
    """
    result = {"root": "above"}
    for node in POSTORDER:
        if node != "root":
            result[node] = "left"
    for leaf in LEAVES:
        result[leaf] = "below"
    return result


SIDE = sides()


def mismatch(first: str, second: str) -> int:
    if first == second:
        return 0
    return 1


def show(value: float) -> str:
    if value == INFINITY:
        return "∞"
    return "%g" % value


def solve() -> "tuple[dict, dict, list]":
    """Sankoff: the score vectors, the winning child symbols, and the working."""
    scores = {}
    winners = {}
    for leaf in LEAVES:
        vector = {}
        for symbol in ALPHABET:
            if symbol == LETTER_OF[leaf]:
                vector[symbol] = 0.0
            else:
                vector[symbol] = INFINITY
        scores[leaf] = vector
        winners[leaf] = {}
    working = []
    for node in POSTORDER:
        scores[node] = {}
        winners[node] = {}
        for symbol in ALPHABET:
            total = 0.0
            picks = {}
            terms = []
            for child in CHILDREN[node]:
                best = INFINITY
                best_symbol = ""
                candidates = []
                for child_symbol in ALPHABET:
                    value = (scores[child][child_symbol]
                             + mismatch(child_symbol, symbol))
                    candidates.append({"symbol": child_symbol,
                                       "score": scores[child][child_symbol],
                                       "cost": mismatch(child_symbol, symbol),
                                       "value": value})
                    if value < best:
                        best = value
                        best_symbol = child_symbol
                total = total + best
                picks[child] = best_symbol
                terms.append({"child": child, "candidates": candidates,
                              "best": best, "best_symbol": best_symbol})
            scores[node][symbol] = total
            winners[node][symbol] = picks
            working.append({"node": node, "symbol": symbol, "terms": terms,
                            "total": total})
    return (scores, winners, working)


SCORES, WINNERS, WORKING = solve()


def cheapest(node: str) -> str:
    best = INFINITY
    chosen = ""
    for symbol in ALPHABET:
        if SCORES[node][symbol] < best:
            best = SCORES[node][symbol]
            chosen = symbol
    return chosen


ROOT_SYMBOL = cheapest("root")
COLUMN_SCORE = SCORES["root"][ROOT_SYMBOL]


def backtrack() -> "list[tuple]":
    """Root first, then each child takes the symbol that gave its parent's score."""
    taken = {"root": ROOT_SYMBOL}
    order = [("root", ROOT_SYMBOL)]
    pending = ["root"]
    while len(pending) > 0:
        node = pending.pop(0)
        for child in CHILDREN[node]:
            if child in LETTER_OF:
                continue
            symbol = WINNERS[node][taken[node]][child]
            taken[child] = symbol
            order.append((child, symbol))
            pending.append(child)
    return order


ASSIGNMENT_ORDER = backtrack()


def full_assignment() -> dict:
    result = {}
    for leaf in LEAVES:
        result[leaf] = LETTER_OF[leaf]
    for node, symbol in ASSIGNMENT_ORDER:
        result[node] = symbol
    return result


ASSIGNMENT = full_assignment()


def mutated_edges() -> "list[tuple]":
    result = []
    for parent, child in EDGES:
        if ASSIGNMENT[parent] != ASSIGNMENT[child]:
            result.append((parent, child))
    return result


MUTATED = mutated_edges()


def brute_force() -> int:
    """Fewest mismatching edges over every assignment to the seven internal nodes."""
    best = 99
    counter = 0
    total_trials = 4 ** len(POSTORDER)
    while counter < total_trials:
        trial = {}
        for leaf in LEAVES:
            trial[leaf] = LETTER_OF[leaf]
        rest = counter
        for node in POSTORDER:
            trial[node] = ALPHABET[rest % 4]
            rest = rest // 4
        total = 0
        for parent, child in EDGES:
            total = total + mismatch(trial[parent], trial[child])
        if total < best:
            best = total
        counter = counter + 1
    return best


def point_segment_distance(px: float, py: float, ax: float, ay: float,
                           bx: float, by: float) -> float:
    """Distance from a point to a segment, used to keep labels off the edges."""
    dx = bx - ax
    dy = by - ay
    squared = dx * dx + dy * dy
    if squared < 1e-12:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    along = ((px - ax) * dx + (py - ay) * dy) / squared
    along = min(max(along, 0.0), 1.0)
    return ((px - (ax + along * dx)) ** 2 + (py - (ay + along * dy)) ** 2) ** 0.5


def vector_left(node: str) -> float:
    """Where a node's row of four scores starts."""
    x, y = POSITION[node]
    if SIDE[node] == "left":
        centre = x - VECTOR_SIDE
    elif SIDE[node] == "right":
        centre = x + VECTOR_SIDE
    else:
        centre = x
    return centre - 1.5 * VECTOR_SPACING


def vector_rows(node: str) -> "tuple[float, float]":
    """The y of the A C G T header and the y of the values beneath it."""
    x, y = POSITION[node]
    if SIDE[node] == "below":
        return (y - LEAF_HEADER_DOWN, y - LEAF_VALUE_DOWN)
    if SIDE[node] == "above":
        return (y + ROOT_HEADER_UP, y + ROOT_VALUE_UP)
    return (y + VECTOR_HEADER_UP, y - VECTOR_VALUE_DOWN)


def verify() -> "list[str]":
    lines = []

    assert len(LEAVES) == 8, "this is the eight-leaf example"
    for node in POSTORDER:
        assert len(CHILDREN[node]) == 2, "the tree must be binary"
    lines.append("balanced rooted tree, %d leaves and %d internal nodes"
                 % (len(LEAVES), len(POSTORDER)))

    for node in POSTORDER:
        got = []
        for symbol in ALPHABET:
            got.append(int(SCORES[node][symbol]))
        assert tuple(got) == PUBLISHED_SCORES[node], (
            "%s computes %s but slide 194 prints %s"
            % (node, tuple(got), PUBLISHED_SCORES[node]))
    lines.append("all %d score-vector entries match slide 194 exactly"
                 % (len(POSTORDER) * len(ALPHABET)))

    assert COLUMN_SCORE == PUBLISHED_SCORE, (
        "the column scores %g but the slide says %d"
        % (COLUMN_SCORE, PUBLISHED_SCORE))
    assert ROOT_SYMBOL == PUBLISHED_ASSIGNMENT["root"], "the root should take C"
    for node in POSTORDER:
        assert ASSIGNMENT[node] == PUBLISHED_ASSIGNMENT[node], (
            "%s is assigned %s but slide 198 prints %s"
            % (node, ASSIGNMENT[node], PUBLISHED_ASSIGNMENT[node]))
    lines.append("the score is %g and all %d ancestral letters match slide 198"
                 % (COLUMN_SCORE, len(POSTORDER)))

    expected = brute_force()
    assert expected == COLUMN_SCORE, (
        "brute force over all %d assignments says %d, not %g"
        % (4 ** len(POSTORDER), expected, COLUMN_SCORE))
    counted = 0
    for parent, child in EDGES:
        counted = counted + mismatch(ASSIGNMENT[parent], ASSIGNMENT[child])
    assert counted == COLUMN_SCORE, "the backtracked assignment misses the optimum"
    lines.append("brute force over all %d internal assignments agrees, and the "
                 "backtracked tree really has %d mutations"
                 % (4 ** len(POSTORDER), counted))

    ties = 0
    for symbol in ALPHABET:
        if SCORES["root"][symbol] == COLUMN_SCORE:
            ties = ties + 1
    assert ties == 1, "the root's cheapest symbol should be unique"
    lines.append("the root's cheapest symbol %s is unique, so nothing is arbitrary"
                 % ROOT_SYMBOL)

    for node in POSTORDER:
        for symbol in ALPHABET:
            rebuilt = 0.0
            for child in CHILDREN[node]:
                winner = WINNERS[node][symbol][child]
                rebuilt = rebuilt + SCORES[child][winner] + mismatch(winner, symbol)
            assert rebuilt == SCORES[node][symbol], (
                "the recorded winners for s(%s, %s) do not rebuild its score"
                % (node, symbol))
    lines.append("every recorded minimum rebuilds the score it belongs to")

    boxes = {}
    for node in POSITION:
        header_y, value_y = vector_rows(node)
        left = vector_left(node) - 0.5 * VECTOR_SPACING
        boxes[node] = (value_y, header_y, left, left + 4 * VECTOR_SPACING)
    for node in boxes:
        for other in boxes:
            if node == other:
                continue
            same_band = (boxes[node][0] < boxes[other][1] + 0.14
                         and boxes[other][0] < boxes[node][1] + 0.14)
            if not same_band:
                continue
            clear = (boxes[node][3] + 0.16 < boxes[other][2]
                     or boxes[other][3] + 0.16 < boxes[node][2])
            assert clear, ("the score rows of %s and %s overlap" % (node, other))
    for node in boxes:
        entry = 0
        while entry < 4:
            x = vector_left(node) + entry * VECTOR_SPACING
            for y in (boxes[node][0], boxes[node][1]):
                for parent, child in EDGES:
                    gap = point_segment_distance(x, y, POSITION[parent][0],
                                                 POSITION[parent][1],
                                                 POSITION[child][0],
                                                 POSITION[child][1])
                    assert gap > 0.15, (
                        "%s's score row sits %.2f in from the %s-%s edge"
                        % (node, gap, parent, child))
            entry = entry + 1
    lines.append("no two score rows overlap and none of them lands on an edge")

    assert POSITION["root"][1] + ROOT_HEADER_UP + 0.25 < CAPTION_Y, (
        "the root's vector runs into the caption")
    assert LEAF_Y - LEAF_VALUE_DOWN - 0.2 > WORK_TOP, (
        "the leaf vectors collide with the working")
    assert WORK_TOP - 3 * WORK_STEP - 0.2 > 0.1, "the working runs off the bottom"
    assert LEFT_MARGIN + (len(LEAVES) - 1) * LEAF_GAP + 0.4 < FIGURE_WIDTH, (
        "the tree runs off the right")
    lines.append("tree spans %.2f to %.2f in, clear of the caption and the working"
                 % (LEFT_MARGIN, LEFT_MARGIN + (len(LEAVES) - 1) * LEAF_GAP))
    return lines


def base_frame(duration: int) -> dict:
    return {"vectors": set(), "ready": {}, "focus_node": "", "focus_symbol": "",
            "lines": [], "assigned": {}, "mutations": [], "caption": "",
            "score_shown": False, "minima": False, "duration_ms": duration}


def candidate_line(term: dict) -> "list[tuple]":
    """min( inf+0, 0+1, inf+1, inf+1 ), with the winning term picked out."""
    pieces = [("min( ", DIM)]
    index = 0
    while index < len(term["candidates"]):
        candidate = term["candidates"][index]
        if index > 0:
            pieces.append((", ", DIM))
        if candidate["symbol"] == term["best_symbol"]:
            colour = GREEN
        else:
            colour = DIM
        pieces.append(("%s+%d" % (show(candidate["score"]), candidate["cost"]),
                       colour))
        index = index + 1
    pieces.append((" )", DIM))
    return pieces


def working_line(item: dict) -> "list[tuple]":
    """One whole line: s(k) = min(...) + min(...) = a + b = total."""
    pieces = [("s(%s)  =  " % item["symbol"], INK)]
    index = 0
    while index < len(item["terms"]):
        if index > 0:
            pieces.append(("  +  ", DIM))
        for piece in candidate_line(item["terms"][index]):
            pieces.append(piece)
        index = index + 1
    pieces.append(("  =  ", DIM))
    index = 0
    while index < len(item["terms"]):
        if index > 0:
            pieces.append((" + ", DIM))
        pieces.append((show(item["terms"][index]["best"]), GREEN))
        index = index + 1
    pieces.append(("  =  ", DIM))
    pieces.append((show(item["total"]), BLUE))
    return pieces


def build_specs() -> "list[dict]":
    specs = []

    opening = base_frame(OPENING_MS)
    opening["caption"] = ("Small parsimony solves one column of the alignment at a "
                          "time, so this is one column on eight species")
    specs.append(opening)

    ready = {}
    leaves = base_frame(LEAF_MS)
    for leaf in LEAVES:
        leaves["vectors"].add(leaf)
        ready[leaf] = list(ALPHABET)
    leaves["ready"] = dict(ready)

    leaves["caption"] = ("A leaf costs nothing for the letter it carries and is "
                         "impossible for the other three")
    specs.append(leaves)

    shown = set(LEAVES)
    lines_so_far = []
    current_node = ""
    for item in WORKING:
        if item["node"] != current_node:
            current_node = item["node"]
            lines_so_far = []
            ready[current_node] = []
        lines_so_far.append(working_line(item))
        ready[current_node].append(item["symbol"])
        if item["node"] == POSTORDER[0]:
            duration = FIRST_SYMBOL_MS
        else:
            duration = SYMBOL_MS
        frame = base_frame(duration)
        frame["vectors"] = set(shown)
        frame["vectors"].add(item["node"])
        frame["ready"] = {}
        for node in ready:
            frame["ready"][node] = list(ready[node])
        frame["focus_node"] = item["node"]
        frame["focus_symbol"] = item["symbol"]
        frame["lines"] = list(lines_so_far)
        frame["caption"] = ("Each child contributes its cheapest option, plus 1 if "
                            "that option differs")
        specs.append(frame)
        if item["symbol"] == ALPHABET[len(ALPHABET) - 1]:
            shown.add(item["node"])

    settled = {}
    for node in ready:
        settled[node] = list(ready[node])

    pick = base_frame(PICK_MS)
    pick["vectors"] = set(shown)
    pick["ready"] = dict(settled)
    pick["focus_node"] = "root"
    pick["focus_symbol"] = ROOT_SYMBOL
    pick["minima"] = True
    pick["score_shown"] = True
    pick["caption"] = ("The root's smallest score is %g, so this column costs %g"
                       % (COLUMN_SCORE, COLUMN_SCORE))
    specs.append(pick)

    assigned = {}
    for node, symbol in ASSIGNMENT_ORDER:
        assigned[node] = symbol
        frame = base_frame(BACK_MS)
        frame["vectors"] = set(shown)
        frame["ready"] = dict(settled)
        frame["assigned"] = dict(assigned)
        frame["focus_node"] = node
        frame["focus_symbol"] = symbol
        frame["minima"] = True
        frame["score_shown"] = True
        if node == "root":
            frame["caption"] = "Backtrack: the root takes %s" % symbol
        else:
            frame["caption"] = ("Each node takes the symbol that gave its parent "
                                "its score: %s here" % symbol)
        specs.append(frame)

    final = base_frame(FINAL_HOLD_MS)
    final["vectors"] = set(shown)
    final["ready"] = dict(settled)
    final["assigned"] = dict(assigned)
    final["mutations"] = list(MUTATED)
    final["minima"] = True
    final["score_shown"] = True
    final["caption"] = ("%d mutations explain this column, and every other column "
                        "is solved the same way" % int(COLUMN_SCORE))
    specs.append(final)
    return specs


def draw_vector(axis: "plt.Axes", node: str, ready: "list[str]", focus: str,
                minima: bool) -> None:
    """A C G T on one row with the four scores directly beneath, deck style."""
    header_y, value_y = vector_rows(node)
    smallest = INFINITY
    for symbol in ready:
        if SCORES[node][symbol] < smallest:
            smallest = SCORES[node][symbol]
    index = 0
    while index < len(ALPHABET):
        symbol = ALPHABET[index]
        x = vector_left(node) + index * VECTOR_SPACING
        if symbol == focus:
            head_colour = BLUE
        else:
            head_colour = INK
        axis.text(x, header_y, symbol, fontsize=HEADER_SIZE, color=head_colour,
                  ha="center", va="center", fontproperties=MONO_BOLD, zorder=6)
        complete = len(ready) == len(ALPHABET)
        if symbol in ready:
            if minima and complete and SCORES[node][symbol] == smallest:
                # Only the root's winning score is picked out, and in the book's
                # green, exactly as Sankoff_step_4.png marks it.
                colour = BOOK_GREEN
                font = MONO_BOLD
            elif symbol == focus:
                colour = BLUE
                font = MONO_BOLD
            else:
                colour = INK
                font = MONO
            text = show(SCORES[node][symbol])
        else:
            colour = FAINT
            font = MONO
            text = "."
        axis.text(x, value_y, text, fontsize=VECTOR_SIZE, color=colour,
                  ha="center", va="center", fontproperties=font, zorder=6)
        index = index + 1


def draw_tree(axis: "plt.Axes", spec: dict) -> None:
    for parent, child in EDGES:
        start = POSITION[parent]
        end = POSITION[child]
        if (parent, child) in spec["mutations"]:
            colour = RED
            width = 2.6
        else:
            colour = INK
            width = 1.7
        axis.plot([start[0], end[0]], [start[1], end[1]], color=colour,
                  linewidth=width, solid_capstyle="round", zorder=3)

    for leaf in LEAVES:
        position = POSITION[leaf]
        disc = patches.Circle(position, LEAF_RADIUS, facecolor=DISC,
                              edgecolor="none", zorder=5)
        axis.add_patch(disc)
        axis.text(position[0], position[1], LETTER_OF[leaf], fontsize=LETTER_SIZE,
                  color="white", ha="center", va="center", fontproperties=MONO_BOLD,
                  zorder=6)
        if leaf in spec["vectors"]:
            draw_vector(axis, leaf, spec["ready"].get(leaf, []), "", False)

    for node in POSTORDER:
        position = POSITION[node]
        if node in spec["assigned"]:
            face = BOOK_GREEN
        else:
            face = BOOK_GREY
        if node == spec["focus_node"] and node not in spec["assigned"]:
            outline = BLUE
            thickness = 2.4
        else:
            outline = face
            thickness = 1.4
        disc = patches.Circle(position, NODE_RADIUS, facecolor=face,
                              edgecolor=outline, linewidth=thickness, zorder=5)
        axis.add_patch(disc)
        if node in spec["assigned"]:
            axis.text(position[0], position[1], spec["assigned"][node],
                      fontsize=LETTER_SIZE - 2, color="white", ha="center",
                      va="center", fontproperties=MONO_BOLD, zorder=6)
        if node in spec["vectors"]:
            if node == spec["focus_node"]:
                focus = spec["focus_symbol"]
            else:
                focus = ""
            draw_vector(axis, node, spec["ready"].get(node, []), focus,
                        spec["minima"] and node == "root")


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)
    draw_tree(axis, spec)

    if len(spec["lines"]) > 0:
        advance = mono_advance(figure, WORK_SIZE)
        index = 0
        while index < len(spec["lines"]):
            pieces = spec["lines"][index]
            total = 0
            for text, colour in pieces:
                total = total + len(text)
            x = FIGURE_WIDTH / 2 - total * advance / 2
            y = WORK_TOP - index * WORK_STEP
            for text, colour in pieces:
                axis.text(x, y, text, fontsize=WORK_SIZE, color=colour, ha="left",
                          va="center", fontproperties=MONO_BOLD, zorder=7)
                x = x + len(text) * advance
            index = index + 1

    if spec["score_shown"]:
        axis.text(SCORE_X, SCORE_Y, "score %g" % COLUMN_SCORE, fontsize=SCORE_SIZE,
                  color=INK, ha="left", va="center", fontproperties=MONO_BOLD,
                  zorder=7)

    if spec["caption"] != "":
        axis.text(FIGURE_WIDTH / 2, CAPTION_Y, spec["caption"], fontsize=LABEL_SIZE,
                  color=INK, ha="center", va="center", fontproperties=OPTIMA_ITALIC,
                  zorder=7)

    figure.savefig(output_path, transparent=True)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: example_small_parsimony.py OUTPUT.gif")
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
    directory = tempfile.mkdtemp(prefix="small_parsimony_")
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
