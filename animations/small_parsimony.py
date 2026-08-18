"""Animated small parsimony, in the 02-180 lecture style.

Uses the four species from Evolutionary_Trees.pptx slide 179 (chimp, human, seal,
whale) on the same balanced tree. Small parsimony solves each column of the
alignment on its own, so this animation follows ONE column all the way through
and shows every number:

  leaves       score 0 for the letter they carry, infinite for the other three
  each node    the four scores s(v, k), one symbol at a time, with the minimum
               over each child's four candidate sums written out in full
  the root     its smallest score is the parsimony score of the column
  backtracking the letter each node takes, chosen by the minimum that produced
               its parent's score, then the one edge that carries a mutation

Every score is checked against a brute-force search over all 64 assignments to
the internal nodes, the backtracked assignment is checked to achieve the minimum,
and the sum over all columns is checked against the score printed on the slide.

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

LEAF_NAMES = ["Chimp", "Human", "Seal", "Whale"]
LEAF_STRINGS = ["ACGTAGGCCT", "ATGTAAGACT", "TCGAGAGCAC", "TCGAAAGCAT"]
ALPHABET = "ACGT"
# The tree of slide 179: root over two internal nodes, each over two leaves.
CHILDREN = {"root": ["left", "right"], "left": ["Chimp", "Human"],
            "right": ["Seal", "Whale"]}
POSTORDER = ["left", "right", "root"]
SLIDE_SCORE = 8
# Position 5 of the alignment (A, A, G, A). Its answer is unique, so nothing has
# to be waved through with "either of these will do".
COLUMN = 4

FIGURE_WIDTH = 12.0
FIGURE_HEIGHT = 7.2
RENDER_DPI = 120
OUTPUT_WIDTH = 1440
OUTPUT_HEIGHT = 864

POSITIONS = {
    "root": (8.10, 6.30),
    "left": (6.08, 4.90), "right": (10.12, 4.90),
    "Chimp": (5.00, 3.40), "Human": (7.15, 3.40),
    "Seal": (9.05, 3.40), "Whale": (11.20, 3.40),
}
# Each node's four scores are written in a row near it. The row is nudged sideways
# so it clears the edge climbing past it, and the entries are all three characters
# wide so nothing has to be measured twice.
VECTOR_SPACING = 0.40
VECTOR_SHIFT = {"left": -0.36, "right": 0.36}
LEAF_RADIUS = 0.21
NODE_RADIUS = 0.16
LETTER_SIZE = 15.0
NAME_SIZE = 11.0
VECTOR_SIZE = 13.0
VECTOR_ABOVE = 0.40
VECTOR_BELOW = 0.72
NAME_BELOW = 0.38

ALIGN_LEFT = 1.70
ALIGN_TOP = 6.45
ALIGN_STEP = 0.36
ALIGN_ADVANCE = 0.165
ALIGN_SIZE = 15.0

CAPTION_Y = 7.00
LABEL_SIZE = 16.0
WORK_TOP = 2.28
WORK_STEP = 0.50
WORK_SIZE = 14.0
SCORE_SIZE = 20.0

OPENING_MS = 3600
LEAF_MS = 3200
SYMBOL_MS = 2800
PICK_MS = 3400
BACK_MS = 2800
FINAL_HOLD_MS = 5600

INFINITY = float("inf")


def leaf_strings() -> dict:
    result = {}
    index = 0
    while index < len(LEAF_NAMES):
        result[LEAF_NAMES[index]] = LEAF_STRINGS[index]
        index = index + 1
    return result


LEAF_OF = leaf_strings()
LENGTH = len(LEAF_STRINGS[0])


def tree_edges() -> "list[tuple]":
    result = []
    for parent in CHILDREN:
        for child in CHILDREN[parent]:
            result.append((parent, child))
    return result


EDGES = tree_edges()


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


def mismatch(first: str, second: str) -> int:
    if first == second:
        return 0
    return 1


def show(value: float) -> str:
    if value == INFINITY:
        return "\u221e"
    return "%g" % value


def leaf_vector(letter: str) -> dict:
    """A leaf costs nothing for the letter it carries and is impossible otherwise."""
    result = {}
    for symbol in ALPHABET:
        if symbol == letter:
            result[symbol] = 0.0
        else:
            result[symbol] = INFINITY
    return result


def solve_column(column: int) -> "tuple[dict, dict, list]":
    """Sankoff on one column: the score vectors, the winners, and the working.

    scores[node][symbol] is the cost of the subtree at node given that node holds
    symbol. winners[node][symbol][child] is the child symbol that achieved the
    minimum, which is exactly what backtracking needs.
    """
    scores = {}
    winners = {}
    for name in LEAF_NAMES:
        scores[name] = leaf_vector(LEAF_OF[name][column])
        winners[name] = {}
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


SCORES, WINNERS, WORKING = solve_column(COLUMN)


def best_symbol_at(node: str, scores: dict) -> str:
    """The cheapest symbol for a node, ties broken alphabetically."""
    best = INFINITY
    chosen = ""
    for symbol in ALPHABET:
        if scores[node][symbol] < best:
            best = scores[node][symbol]
            chosen = symbol
    return chosen


ROOT_SYMBOL = best_symbol_at("root", SCORES)
COLUMN_SCORE = SCORES["root"][ROOT_SYMBOL]


def backtrack() -> "list[tuple]":
    """Root first, then each child takes the symbol that produced its parent's score."""
    order = [("root", ROOT_SYMBOL)]
    pending = ["root"]
    while len(pending) > 0:
        node = pending.pop(0)
        symbol = ""
        for name, taken in order:
            if name == node:
                symbol = taken
        for child in CHILDREN[node]:
            if child in LEAF_OF:
                continue
            order.append((child, WINNERS[node][symbol][child]))
            pending.append(child)
    return order


ASSIGNMENT_ORDER = backtrack()


def full_assignment() -> dict:
    result = {}
    for name in LEAF_NAMES:
        result[name] = LEAF_OF[name][COLUMN]
    for node, symbol in ASSIGNMENT_ORDER:
        result[node] = symbol
    return result


ASSIGNMENT = full_assignment()


def mutated_edges(assignment: dict) -> "list[tuple]":
    result = []
    for parent, child in EDGES:
        if assignment[parent] != assignment[child]:
            result.append((parent, child))
    return result


MUTATED = mutated_edges(ASSIGNMENT)


def brute_force(column: int) -> int:
    """Smallest number of mismatching edges over all assignments to three nodes."""
    best = 99
    for first in ALPHABET:
        for second in ALPHABET:
            for third in ALPHABET:
                trial = {"root": first, "left": second, "right": third}
                for name in LEAF_NAMES:
                    trial[name] = LEAF_OF[name][column]
                total = 0
                for parent, child in EDGES:
                    total = total + mismatch(trial[parent], trial[child])
                if total < best:
                    best = total
    return best


def verify() -> "list[str]":
    lines = []

    for name in LEAF_NAMES:
        assert len(LEAF_OF[name]) == LENGTH, "the strings must be aligned"
    lines.append("%d aligned strings of length %d, from slide 179"
                 % (len(LEAF_NAMES), LENGTH))

    column_total = 0
    position = 0
    while position < LENGTH:
        scores, winners, working = solve_column(position)
        chosen = best_symbol_at("root", scores)
        value = scores["root"][chosen]
        expected = brute_force(position)
        assert value == expected, (
            "position %d: the dynamic program says %g but brute force over all 64 "
            "assignments says %d" % (position, value, expected))
        column_total = column_total + int(value)
        position = position + 1
    lines.append("every one of the %d columns agrees with brute force over all 64 "
                 "assignments" % LENGTH)

    assert column_total == SLIDE_SCORE, (
        "the columns total %d but slide 179 prints %d" % (column_total, SLIDE_SCORE))
    lines.append("the columns total %d, matching slide 179" % column_total)

    # The animation walks one column, so that column has to be right on its own.
    assert COLUMN_SCORE == brute_force(COLUMN), "the animated column disagrees"
    counted = 0
    for parent, child in EDGES:
        counted = counted + mismatch(ASSIGNMENT[parent], ASSIGNMENT[child])
    assert counted == COLUMN_SCORE, (
        "backtracking produced %d mutations but the score is %g"
        % (counted, COLUMN_SCORE))
    lines.append("position %d scores %g, and the backtracked assignment achieves it "
                 "with %d mutation" % (COLUMN + 1, COLUMN_SCORE, counted))

    # A unique answer keeps the backtracking honest: no arbitrary tie-breaking.
    ties = 0
    for symbol in ALPHABET:
        if SCORES["root"][symbol] == COLUMN_SCORE:
            ties = ties + 1
    assert ties == 1, "the root should have a unique cheapest symbol, found %d" % ties
    lines.append("the root's cheapest symbol %s is unique, so nothing is arbitrary"
                 % ROOT_SYMBOL)

    for node in POSTORDER:
        for symbol in ALPHABET:
            rebuilt = 0.0
            for child in CHILDREN[node]:
                winner = WINNERS[node][symbol][child]
                rebuilt = rebuilt + (SCORES[child][winner]
                                     + mismatch(winner, symbol))
            assert rebuilt == SCORES[node][symbol], (
                "the recorded winners for s(%s, %s) do not rebuild its score"
                % (node, symbol))
    lines.append("every recorded minimum really rebuilds the score it belongs to")

    for name in POSITIONS:
        for other in POSITIONS:
            if name == other:
                continue
            if abs(POSITIONS[name][1] - POSITIONS[other][1]) < 0.01:
                gap = abs(POSITIONS[name][0] - POSITIONS[other][0])
                assert gap > 1.3, ("%s and %s are only %.2f in apart on one row"
                                   % (name, other, gap))
    rows = {}
    for node in POSITIONS:
        if node in POSTORDER:
            y = POSITIONS[node][1] + VECTOR_ABOVE
        else:
            y = POSITIONS[node][1] - VECTOR_BELOW
        left = vector_left(node) - 0.5 * VECTOR_SPACING
        rows[node] = (y, left, left + 4 * VECTOR_SPACING)
    for node in rows:
        for other in rows:
            if node == other:
                continue
            if abs(rows[node][0] - rows[other][0]) > 0.25:
                continue
            clear = (rows[node][2] + 0.25 < rows[other][1]
                     or rows[other][2] + 0.25 < rows[node][1])
            assert clear, ("the score rows of %s and %s are not 0.25 in apart"
                           % (node, other))
    for node in rows:
        y = rows[node][0]
        entry = 0
        while entry < 4:
            x = vector_left(node) + entry * VECTOR_SPACING
            for parent, child in EDGES:
                gap = point_segment_distance(x, y, POSITIONS[parent][0],
                                             POSITIONS[parent][1],
                                             POSITIONS[child][0],
                                             POSITIONS[child][1])
                assert gap > 0.16, ("%s's score row sits %.2f in from the %s-%s edge"
                                    % (node, gap, parent, child))
            entry = entry + 1
    lines.append("no two rows of scores overlap, and none of them lands on an edge")

    assert POSITIONS["root"][1] + VECTOR_ABOVE + 0.2 < CAPTION_Y, (
        "the root's vector runs into the caption")
    assert POSITIONS["Chimp"][1] - VECTOR_BELOW - 0.2 > WORK_TOP, (
        "the leaf vectors collide with the working")
    assert WORK_TOP - 2 * WORK_STEP - 0.3 > 0.3, "the working runs off the bottom"
    assert ALIGN_LEFT + LENGTH * ALIGN_ADVANCE < POSITIONS["Chimp"][0] - 1.0, (
        "the alignment collides with the tree")
    lines.append("tree, alignment and working all sit clear of each other")
    return lines


def base_frame(duration: int) -> dict:
    return {"vectors": set(), "focus_node": "", "focus_symbol": "", "lines": [],
            "assigned": {}, "mutations": [], "caption": "", "score_shown": False,
            "closing": "", "duration_ms": duration}


def candidate_line(term: dict, symbol: str) -> "list[tuple]":
    """min( 0+0, inf+1, inf+1, inf+1 ) = 0, with the winning term picked out."""
    pieces = [("from %-5s  " % term["child"], INK), ("min( ", DIM)]
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
    pieces.append((" )  =  ", DIM))
    pieces.append((show(term["best"]), GREEN))
    return pieces


def build_specs() -> "list[dict]":
    specs = []

    opening = base_frame(OPENING_MS)
    opening["caption"] = ("Small parsimony solves one column at a time, so follow "
                          "position %d" % (COLUMN + 1))
    specs.append(opening)

    leaves = base_frame(LEAF_MS)
    for name in LEAF_NAMES:
        leaves["vectors"].add(name)
    leaves["caption"] = ("A leaf costs nothing for the letter it carries, and is "
                         "impossible for the other three")
    specs.append(leaves)

    shown = set()
    for name in LEAF_NAMES:
        shown.add(name)

    for item in WORKING:
        frame = base_frame(SYMBOL_MS)
        frame["vectors"] = set(shown)
        frame["vectors"].add(item["node"])
        frame["focus_node"] = item["node"]
        frame["focus_symbol"] = item["symbol"]
        rows = []
        for term in item["terms"]:
            rows.append(candidate_line(term, item["symbol"]))
        summary = [("s(%s, %s)  =  " % (item["node"], item["symbol"]), INK)]
        index = 0
        while index < len(item["terms"]):
            if index > 0:
                summary.append(("  +  ", DIM))
            summary.append((show(item["terms"][index]["best"]), GREEN))
            index = index + 1
        summary.append(("  =  ", DIM))
        summary.append((show(item["total"]), BLUE))
        rows.append(summary)
        frame["lines"] = rows
        frame["caption"] = ("Each child contributes its cheapest option, plus 1 if "
                            "that option differs")
        specs.append(frame)
        if item["symbol"] == ALPHABET[len(ALPHABET) - 1]:
            shown.add(item["node"])

    pick = base_frame(PICK_MS)
    pick["vectors"] = set(shown)
    pick["focus_node"] = "root"
    pick["focus_symbol"] = ROOT_SYMBOL
    pick["score_shown"] = True
    pick["caption"] = ("The root's smallest score is %g, so this column costs %g"
                       % (COLUMN_SCORE, COLUMN_SCORE))
    specs.append(pick)

    assigned = {}
    for node, symbol in ASSIGNMENT_ORDER:
        assigned[node] = symbol
        frame = base_frame(BACK_MS)
        frame["vectors"] = set(shown)
        frame["assigned"] = dict(assigned)
        frame["focus_node"] = node
        frame["focus_symbol"] = symbol
        frame["score_shown"] = True
        if node == "root":
            frame["caption"] = "Backtrack: the root takes %s" % symbol
        else:
            parent = ""
            for candidate in CHILDREN:
                if node in CHILDREN[candidate]:
                    parent = candidate
            frame["caption"] = ("%s takes the symbol that gave %s its score: %s"
                                % (node, parent, symbol))
        specs.append(frame)

    final = base_frame(FINAL_HOLD_MS)
    final["vectors"] = set(shown)
    final["assigned"] = dict(assigned)
    final["mutations"] = list(MUTATED)
    final["score_shown"] = True
    final["caption"] = ("One mutation explains position %d, and the other columns "
                        "add up the same way" % (COLUMN + 1))
    final["closing"] = ("The whole alignment needs %d mutations" % SLIDE_SCORE)
    specs.append(final)
    return specs


def draw_alignment(axis: "plt.Axes") -> None:
    """The four strings, with the animated column boxed."""
    index = 0
    while index < len(LEAF_NAMES):
        name = LEAF_NAMES[index]
        y = ALIGN_TOP - index * ALIGN_STEP
        axis.text(ALIGN_LEFT - 0.16, y, name, fontsize=NAME_SIZE, color=DIM,
                  ha="right", va="center", fontproperties=OPTIMA, zorder=5)
        position = 0
        while position < LENGTH:
            if position == COLUMN:
                colour = INK
                font = MONO_BOLD
            else:
                colour = FAINT
                font = MONO
            axis.text(ALIGN_LEFT + (position + 0.5) * ALIGN_ADVANCE, y,
                      LEAF_OF[name][position], fontsize=ALIGN_SIZE, color=colour,
                      ha="center", va="center", fontproperties=font, zorder=5)
            position = position + 1
        index = index + 1
    box = patches.Rectangle(
        (ALIGN_LEFT + COLUMN * ALIGN_ADVANCE - 0.01,
         ALIGN_TOP - (len(LEAF_NAMES) - 1) * ALIGN_STEP - 0.17),
        ALIGN_ADVANCE + 0.02, (len(LEAF_NAMES) - 1) * ALIGN_STEP + 0.34,
        facecolor="none", edgecolor=BLUE, linewidth=1.2, zorder=6)
    axis.add_patch(box)


def vector_left(node: str) -> float:
    """Where a node's row of scores starts."""
    return (POSITIONS[node][0] + VECTOR_SHIFT.get(node, 0.0)
            - 1.5 * VECTOR_SPACING)


def draw_partial_vector(axis: "plt.Axes", node: str, y: float, ready: "list[str]",
                        focus_symbol: str) -> None:
    """A node's four scores, with the ones not computed yet left as dots."""
    x = vector_left(node)
    for symbol in ALPHABET:
        if symbol in ready:
            if symbol == focus_symbol:
                colour = BLUE
                font = MONO_BOLD
            else:
                colour = DIM
                font = MONO
            text = "%s %s" % (symbol, show(SCORES[node][symbol]))
        else:
            colour = FAINT
            font = MONO
            text = "%s ." % symbol
        axis.text(x, y, text, fontsize=VECTOR_SIZE, color=colour, ha="center",
                  va="center", fontproperties=font, zorder=6)
        x = x + VECTOR_SPACING


def symbols_ready(node: str, focus_node: str, focus_symbol: str) -> "list[str]":
    """Which of a node's four scores have been computed by this frame."""
    if node != focus_node:
        return list(ALPHABET)
    result = []
    for symbol in ALPHABET:
        result.append(symbol)
        if symbol == focus_symbol:
            return result
    return result


def draw_tree(axis: "plt.Axes", spec: dict) -> None:
    for parent, child in EDGES:
        start = POSITIONS[parent]
        end = POSITIONS[child]
        if (parent, child) in spec["mutations"]:
            colour = RED
            width = 2.6
        else:
            colour = INK
            width = 1.8
        axis.plot([start[0], end[0]], [start[1], end[1]], color=colour,
                  linewidth=width, solid_capstyle="round", zorder=3)

    for name in LEAF_NAMES:
        position = POSITIONS[name]
        disc = patches.Circle(position, LEAF_RADIUS, facecolor=DISC,
                              edgecolor="none", zorder=5)
        axis.add_patch(disc)
        axis.text(position[0], position[1], LEAF_OF[name][COLUMN],
                  fontsize=LETTER_SIZE, color="white", ha="center", va="center",
                  fontproperties=MONO_BOLD, zorder=6)
        axis.text(position[0], position[1] - NAME_BELOW, name, fontsize=NAME_SIZE,
                  color=DIM, ha="center", va="center", fontproperties=OPTIMA,
                  zorder=6)
        if name in spec["vectors"]:
            draw_partial_vector(axis, name, position[1] - VECTOR_BELOW,
                                list(ALPHABET), "")

    for node in POSTORDER:
        position = POSITIONS[node]
        if node in spec["assigned"]:
            face = INK
        else:
            face = "#FFFFFF"
        disc = patches.Circle(position, NODE_RADIUS, facecolor=face,
                              edgecolor=INK, linewidth=1.4, zorder=5)
        axis.add_patch(disc)
        if node in spec["assigned"]:
            axis.text(position[0], position[1], spec["assigned"][node],
                      fontsize=LETTER_SIZE - 2, color="white", ha="center",
                      va="center", fontproperties=MONO_BOLD, zorder=6)
        if node in spec["vectors"]:
            ready = symbols_ready(node, spec["focus_node"], spec["focus_symbol"])
            if node == spec["focus_node"]:
                focus = spec["focus_symbol"]
            else:
                focus = ""
            draw_partial_vector(axis, node, position[1] + VECTOR_ABOVE, ready, focus)


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)
    draw_alignment(axis)
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
        axis.text(ALIGN_LEFT + LENGTH * ALIGN_ADVANCE / 2,
                  ALIGN_TOP - len(LEAF_NAMES) * ALIGN_STEP - 0.45,
                  "score %g" % COLUMN_SCORE, fontsize=SCORE_SIZE, color=INK,
                  ha="center", va="center", fontproperties=MONO_BOLD, zorder=7)

    if spec["closing"] != "":
        axis.text(FIGURE_WIDTH / 2, WORK_TOP - 0.1, spec["closing"],
                  fontsize=LABEL_SIZE, color=INK, ha="center", va="center",
                  fontproperties=OPTIMA_ITALIC, zorder=7)

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
