"""Animated neighbor-joining, in the 02-180 lecture style.

Follows Evolutionary_Trees.pptx slides 145-171. Both matrices are on screen at
once: D with its TotalDistance column on the left, D* on the right, filling in
one entry at a time. Every number is worked out in the panel underneath, so the
viewer sees where it comes from:

  TotalDistance(i)   one taxon at a time, as the sum of its row
  D*(i,j)            one entry at a time, from D(i,j) and the two totals
  limb lengths       from delta = (TotalDistance(i) - TotalDistance(j))/(n-2)
  the reduced D      each new entry from the three old distances it replaces

With three taxa left every pair is a neighbor, which is the base case.

The additive matrix is recovered from the deck (its TotalDistance column pins it
down), and every intermediate matrix and limb length is asserted against the
numbers printed on the slides. The finished tree is also checked to reproduce D
exactly, which is the whole point of neighbor-joining on an additive matrix.

Run:  python3 example_neighbor_joining.py OUTPUT.gif
"""

import os
import sys
import tempfile

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, BLUE, DIM, DISC, FAINT, GREEN, INK, MONO,
                           MONO_BOLD, RED, OPTIMA_ITALIC, ease, mono_advance,
                           new_axes)
from make_gif import assemble_transparent_gif

TAXA = ["v1", "v2", "v3", "v4"]
DISTANCES = [
    [0, 13, 21, 22],
    [13, 0, 12, 13],
    [21, 12, 0, 13],
    [22, 13, 13, 0],
]
# Printed on slide 145.
PUBLISHED_TOTAL = [56, 38, 46, 48]
PUBLISHED_STAR = [
    [0, -68, -60, -60],
    [-68, 0, -60, -60],
    [-60, -60, 0, -68],
    [-60, -60, -68, 0],
]
# Printed on slide 171.
PUBLISHED_LIMBS = {"v1": 11, "v2": 2, "v3": 6, "v4": 7, "m": 4}

FIGURE_WIDTH = 12.6
FIGURE_HEIGHT = 6.5
RENDER_DPI = 115
OUTPUT_WIDTH = 1449
OUTPUT_HEIGHT = 748

CELL = 0.62
D_LEFT = 0.45
STAR_LEFT = 4.62
MATRIX_TOP = 5.55
HEADER_SIZE = 13.0
VALUE_SIZE = 14.0
TITLE_SIZE = 17.0
TOTAL_GAP = 1.6

LEAF_RADIUS = 0.19
LEAF_SIZE = 11.0
NODE_RADIUS = 0.10
LIMB_SIZE = 13.0

CAPTION_Y = 6.18
LABEL_SIZE = 16.0
# The panel where every number is worked out, under both matrices.
WORK_TOP = 2.45
WORK_STEP = 0.52
WORK_SIZE = 16.0

TOTAL_MS = 1600
STAR_MS = 1600
# The second round repeats moves the viewer has already seen, so it runs quicker.
REPEAT_SCALE = 0.7
HOLD_MS = 2700
LIMB_MS = 2500
NEW_MS = 2400
GROW_FRAMES = 9
GROW_MS = 110
FINAL_HOLD_MS = 5600

# Schematic tree positions, laid out like slide 171.
POSITIONS = {
    "v1": (9.35, 5.40), "v2": (9.35, 3.45),
    "m": (10.15, 4.43), "c": (11.05, 4.43),
    "v3": (11.85, 5.40), "v4": (11.85, 3.45),
}
TREE_EDGES = [("v1", "m"), ("v2", "m"), ("m", "c"), ("v3", "c"), ("v4", "c")]


def total_distance(names: "list[str]", matrix: dict) -> dict:
    result = {}
    for name in names:
        total = 0.0
        for other in names:
            total = total + matrix[(name, other)]
        result[name] = total
    return result


def neighbor_matrix(names: "list[str]", matrix: dict) -> dict:
    """D*(i,j) = (n-2) D(i,j) - TotalDistance(i) - TotalDistance(j)."""
    totals = total_distance(names, matrix)
    count = len(names)
    result = {}
    for first in names:
        for second in names:
            if first == second:
                result[(first, second)] = 0.0
            else:
                result[(first, second)] = ((count - 2) * matrix[(first, second)]
                                           - totals[first] - totals[second])
    return result


def run_neighbor_joining() -> "list[dict]":
    """Record each round: D, D*, the joined pair, limbs, and the reduced matrix."""
    names = list(TAXA)
    matrix = {}
    row = 0
    while row < len(TAXA):
        column = 0
        while column < len(TAXA):
            matrix[(TAXA[row], TAXA[column])] = float(DISTANCES[row][column])
            column = column + 1
        row = row + 1

    rounds = []
    while len(names) > 3:
        totals = total_distance(names, matrix)
        star = neighbor_matrix(names, matrix)
        before = dict(matrix)
        best = None
        first_index = 0
        while first_index < len(names):
            second_index = first_index + 1
            while second_index < len(names):
                value = star[(names[first_index], names[second_index])]
                if best is None or value < best[0]:
                    best = (value, names[first_index], names[second_index])
                second_index = second_index + 1
            first_index = first_index + 1
        value, left, right = best
        count = len(names)
        delta = (totals[left] - totals[right]) / (count - 2)
        limb_left = (matrix[(left, right)] + delta) / 2
        limb_right = (matrix[(left, right)] - delta) / 2

        reduced = []
        for name in names:
            if name != left and name != right:
                reduced.append(name)
        new_name = "m"
        replacements = []
        for name in reduced:
            joined = (matrix[(left, name)] + matrix[(right, name)]
                      - matrix[(left, right)]) / 2
            matrix[(new_name, name)] = joined
            matrix[(name, new_name)] = joined
            replacements.append({"other": name, "from_left": before[(left, name)],
                                 "from_right": before[(right, name)],
                                 "across": before[(left, right)], "value": joined})
        matrix[(new_name, new_name)] = 0.0
        reduced.append(new_name)

        rounds.append({"names": list(names), "matrix": before, "best": value,
                       "totals": dict(totals), "star": dict(star),
                       "pair": (left, right), "delta": delta, "new_name": new_name,
                       "limbs": {left: limb_left, right: limb_right},
                       "replacements": replacements,
                       "after_names": list(reduced),
                       "after_matrix": dict(matrix)})
        names = reduced

    # Base case: three taxa, every pair a neighbor, so hang a star.
    totals = total_distance(names, matrix)
    star = neighbor_matrix(names, matrix)
    limbs = {}
    sources = {}
    for name in names:
        others = []
        for other in names:
            if other != name:
                others.append(other)
        limbs[name] = (matrix[(name, others[0])] + matrix[(name, others[1])]
                       - matrix[(others[0], others[1])]) / 2
        sources[name] = (others[0], others[1])
    rounds.append({"names": list(names), "matrix": dict(matrix), "best": 0.0,
                   "totals": dict(totals), "star": dict(star), "pair": (),
                   "delta": 0.0, "new_name": "", "limbs": dict(limbs),
                   "sources": sources, "replacements": [], "after_names": [],
                   "after_matrix": dict(matrix)})
    return rounds


ROUNDS = run_neighbor_joining()


def all_limbs() -> dict:
    result = {}
    for entry in ROUNDS:
        for name in entry["limbs"]:
            result[name] = entry["limbs"][name]
    return result


LIMBS = all_limbs()


def edge_lengths() -> dict:
    """Length of each tree edge, keyed both ways for easy lookup."""
    result = {}
    for a, b in TREE_EDGES:
        if a == "m" and b == "c":
            length = LIMBS["m"]
        elif a in LIMBS:
            length = LIMBS[a]
        else:
            length = LIMBS[b]
        result[(a, b)] = length
        result[(b, a)] = length
    return result


EDGE_LENGTH = edge_lengths()


def tree_distance(start: str, goal: str) -> float:
    """Path length between two nodes of the finished tree."""
    neighbours = {}
    for a, b in TREE_EDGES:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    stack = [(start, 0.0, "")]
    while len(stack) > 0:
        node, distance, came = stack.pop()
        if node == goal:
            return distance
        for other in neighbours[node]:
            if other == came:
                continue
            stack.append((other, distance + EDGE_LENGTH[(node, other)], node))
    raise ValueError("no path from %s to %s" % (start, goal))


def upper_pairs(names: "list[str]") -> "list[tuple]":
    """Each unordered pair once, in reading order across the matrix."""
    result = []
    first = 0
    while first < len(names):
        second = first + 1
        while second < len(names):
            result.append((names[first], names[second]))
            second = second + 1
        first = first + 1
    return result


def verify() -> "list[str]":
    lines = []
    size = len(TAXA)

    first = ROUNDS[0]
    totals = []
    for name in TAXA:
        totals.append(first["totals"][name])
    assert totals == [float(value) for value in PUBLISHED_TOTAL], (
        "TotalDistance is %s but slide 145 prints %s" % (totals, PUBLISHED_TOTAL))
    lines.append("TotalDistance is %s, matching slide 145"
                 % " ".join("%g" % value for value in totals))

    # Each total must really be the sum of its own row, which is what the
    # animation shows one taxon at a time.
    for name in TAXA:
        running = 0.0
        for other in TAXA:
            running = running + first["matrix"][(name, other)]
        assert running == first["totals"][name], "row sum for %s disagrees" % name
    lines.append("every total is the sum of its own row, term by term")

    row = 0
    while row < size:
        column = 0
        while column < size:
            got = first["star"][(TAXA[row], TAXA[column])]
            assert got == float(PUBLISHED_STAR[row][column]), (
                "D*(%s,%s) is %g but the slide prints %g"
                % (TAXA[row], TAXA[column], got, PUBLISHED_STAR[row][column]))
            column = column + 1
        row = row + 1
    lines.append("the whole neighbor-joining matrix D* matches slide 145")

    assert first["pair"] == ("v1", "v2"), (
        "expected v1 and v2 to join first, got %s" % (first["pair"],))
    assert first["delta"] == 9, "delta should be 9, got %g" % first["delta"]
    lines.append("v1 and v2 join first at D* = -68, with delta = 9")

    # The limb lengths must satisfy the pair of equations they come from:
    # they add up to D(i,j) and differ by delta.
    left, right = first["pair"]
    assert (first["limbs"][left] + first["limbs"][right]
            == first["matrix"][(left, right)]), "limbs must add up to D(i,j)"
    assert first["limbs"][left] - first["limbs"][right] == first["delta"], (
        "limbs must differ by delta")
    lines.append("the two limbs add to D(v1,v2) = 13 and differ by delta = 9")

    reduced = ROUNDS[1]
    assert sorted(reduced["names"]) == ["m", "v3", "v4"], (
        "the reduced matrix should be on m, v3, v4")
    assert reduced["matrix"][("m", "v3")] == 10, "D'(m,v3) should be 10"
    assert reduced["matrix"][("m", "v4")] == 11, "D'(m,v4) should be 11"
    lines.append("the reduced matrix has D'(m,v3) = 10 and D'(m,v4) = 11")

    values = []
    for name in reduced["names"]:
        for other in reduced["names"]:
            if name != other:
                values.append(reduced["star"][(name, other)])
    assert len(set(values)) == 1, "with three taxa every D* entry should be equal"
    lines.append("with three taxa left every D* entry is %g: all of them are neighbors"
                 % values[0])

    for name in PUBLISHED_LIMBS:
        assert LIMBS[name] == PUBLISHED_LIMBS[name], (
            "limb %s is %g but slide 171 prints %g"
            % (name, LIMBS[name], PUBLISHED_LIMBS[name]))
    lines.append("all five limb lengths match slide 171: %s"
                 % " ".join("%s=%g" % (name, LIMBS[name])
                            for name in ["v1", "v2", "v3", "v4", "m"]))

    row = 0
    while row < size:
        column = 0
        while column < size:
            if row != column:
                through = tree_distance(TAXA[row], TAXA[column])
                assert abs(through - DISTANCES[row][column]) < 1e-9, (
                    "tree gives d(%s,%s) = %g but D says %g"
                    % (TAXA[row], TAXA[column], through, DISTANCES[row][column]))
            column = column + 1
        row = row + 1
    lines.append("the finished tree reproduces every entry of D exactly")

    d_right = D_LEFT + (size + TOTAL_GAP + 0.5) * CELL
    assert d_right < STAR_LEFT, "D collides with D*"
    star_right = STAR_LEFT + (size + 2.0) * CELL
    assert star_right < POSITIONS["v1"][0] - 0.4, "D* collides with the tree"
    assert POSITIONS["v3"][0] + 0.35 < FIGURE_WIDTH, "tree runs off the right"
    assert POSITIONS["v1"][1] + 0.35 < CAPTION_Y, "tree runs into the caption"
    lowest = MATRIX_TOP - size * CELL
    assert lowest - 0.3 > WORK_TOP, "the matrices collide with the working"
    assert POSITIONS["v2"][1] - LEAF_RADIUS > WORK_TOP + 0.2, (
        "the tree collides with the working")
    assert WORK_TOP - 3 * WORK_STEP > 0.4, "the working runs off the bottom"
    lines.append("D ends at %.2f in, D* spans %.2f to %.2f in, tree starts at %.2f in"
                 % (d_right, STAR_LEFT, star_right, POSITIONS["v1"][0]))
    return lines


def paced(duration: int, round_index: int) -> int:
    """Rounds after the first are repetition, so they get less dwell."""
    if round_index == 0:
        return duration
    return int(round(duration * REPEAT_SCALE / 10.0)) * 10


def base_frame(duration: int) -> dict:
    return {"names": [], "values": {}, "totals": {}, "totals_shown": set(),
            "d_title": "D", "star_title": "D*", "star_values": {},
            "star_shown": set(), "row_focus": "", "cell_focus": set(),
            "star_focus": (), "pair": (), "lines": [], "built": [], "growth": 1.0,
            "caption": "", "duration_ms": duration}


def round_frame(entry: dict, duration: int) -> dict:
    """A frame carrying one round's two matrices, with nothing revealed yet."""
    frame = base_frame(duration)
    frame["names"] = list(entry["names"])
    frame["values"] = dict(entry["matrix"])
    frame["totals"] = dict(entry["totals"])
    frame["star_values"] = dict(entry["star"])
    return frame


def number(value: float) -> str:
    return "%g" % value


def total_line(entry: dict, name: str) -> "list[tuple]":
    """TotalDistance(v1) = 0 + 13 + 21 + 22 = 56, as coloured pieces."""
    pieces = [("TotalDistance(%s)  =  " % name, INK)]
    index = 0
    while index < len(entry["names"]):
        other = entry["names"][index]
        if index > 0:
            pieces.append((" + ", DIM))
        pieces.append((number(entry["matrix"][(name, other)]), BLUE))
        index = index + 1
    pieces.append(("  =  ", DIM))
    pieces.append((number(entry["totals"][name]), GREEN))
    return pieces


def star_line(entry: dict, first: str, second: str) -> "list[tuple]":
    """D*(v1,v2) = 2 · 13 - 56 - 38 = -68."""
    count = len(entry["names"])
    pieces = [("D*(%s,%s)  =  " % (first, second), INK),
              ("%d" % (count - 2), DIM),
              (" · ", DIM),
              (number(entry["matrix"][(first, second)]), BLUE),
              ("  -  ", DIM),
              (number(entry["totals"][first]), RED),
              ("  -  ", DIM),
              (number(entry["totals"][second]), RED),
              ("  =  ", DIM),
              (number(entry["star"][(first, second)]), GREEN)]
    return pieces


def build_specs() -> "list[dict]":
    specs = []
    built = []

    round_index = 0
    while round_index < len(ROUNDS):
        entry = ROUNDS[round_index]
        names = entry["names"]
        count = len(names)
        if round_index == 0:
            opening_caption = "The distance matrix D, and the empty matrix D* beside it"
        else:
            opening_caption = "Start again on the reduced matrix"

        opening = round_frame(entry, HOLD_MS)
        opening["built"] = list(built)
        opening["caption"] = opening_caption
        if round_index > 0:
            opening["d_title"] = "D'"
            opening["star_title"] = "D'*"
        specs.append(opening)

        # TotalDistance, one taxon at a time, as the sum of its own row.
        shown_totals = set()
        for name in names:
            shown_totals.add(name)
            frame = round_frame(entry, paced(TOTAL_MS, round_index))
            frame["built"] = list(built)
            frame["totals_shown"] = set(shown_totals)
            frame["row_focus"] = name
            frame["lines"] = [total_line(entry, name)]
            frame["caption"] = "TotalDistance of a taxon is the sum of its row"
            if round_index > 0:
                frame["d_title"] = "D'"
                frame["star_title"] = "D'*"
            specs.append(frame)

        # D*, one entry at a time, from D and the two totals.
        shown_star = set()
        for first, second in upper_pairs(names):
            shown_star.add((first, second))
            shown_star.add((second, first))
            frame = round_frame(entry, paced(STAR_MS, round_index))
            frame["built"] = list(built)
            frame["totals_shown"] = set(shown_totals)
            frame["star_shown"] = set(shown_star)
            frame["cell_focus"] = set([(first, second), (second, first)])
            frame["star_focus"] = (first, second)
            frame["row_focus"] = ""
            frame["lines"] = [star_line(entry, first, second)]
            frame["caption"] = ("D*(i,j) = (n-2) D(i,j) - TotalDistance(i) "
                                "- TotalDistance(j)")
            if round_index > 0:
                frame["d_title"] = "D'"
                frame["star_title"] = "D'*"
            specs.append(frame)

        full_star = set()
        for first, second in upper_pairs(names):
            full_star.add((first, second))
            full_star.add((second, first))

        if len(entry["pair"]) == 2:
            left, right = entry["pair"]

            pick = round_frame(entry, HOLD_MS)
            pick["built"] = list(built)
            pick["totals_shown"] = set(shown_totals)
            pick["star_shown"] = set(full_star)
            pick["pair"] = entry["pair"]
            pick["caption"] = ("The smallest entry of D* is %s, so %s and %s are "
                               "neighbors" % (number(entry["best"]), left, right))
            if round_index > 0:
                pick["d_title"] = "D'"
                pick["star_title"] = "D'*"
            specs.append(pick)

            delta_lines = [
                [("delta  =  ( TotalDistance(%s) - TotalDistance(%s) ) / (n - 2)  =  "
                  % (left, right), INK),
                 ("( %s - %s ) / %d" % (number(entry["totals"][left]),
                                        number(entry["totals"][right]), count - 2), RED),
                 ("  =  ", DIM),
                 (number(entry["delta"]), GREEN)],
                [("limb(%s)  =  ( D(%s,%s) + delta ) / 2  =  ( %s + %s ) / 2  =  "
                  % (left, left, right, number(entry["matrix"][(left, right)]),
                     number(entry["delta"])), INK),
                 (number(entry["limbs"][left]), GREEN)],
                [("limb(%s)  =  ( D(%s,%s) - delta ) / 2  =  ( %s - %s ) / 2  =  "
                  % (right, left, right, number(entry["matrix"][(left, right)]),
                     number(entry["delta"])), INK),
                 (number(entry["limbs"][right]), GREEN)],
            ]

            shown_lines = 1
            while shown_lines <= 3:
                frame = round_frame(entry, LIMB_MS)
                frame["built"] = list(built)
                frame["totals_shown"] = set(shown_totals)
                frame["star_shown"] = set(full_star)
                frame["pair"] = entry["pair"]
                frame["lines"] = delta_lines[0:shown_lines]
                frame["caption"] = "The two limb lengths follow from delta"
                if round_index > 0:
                    frame["d_title"] = "D'"
                    frame["star_title"] = "D'*"
                specs.append(frame)
                shown_lines = shown_lines + 1

            # Hang the two limbs on the tree.
            growing = list(built)
            growing.append((left, entry["new_name"]))
            growing.append((right, entry["new_name"]))
            step = 1
            while step <= GROW_FRAMES:
                frame = round_frame(entry, GROW_MS)
                frame["built"] = list(growing)
                frame["totals_shown"] = set(shown_totals)
                frame["star_shown"] = set(full_star)
                frame["pair"] = entry["pair"]
                frame["growth"] = ease(step / GROW_FRAMES)
                frame["lines"] = delta_lines[1:3]
                frame["caption"] = ("Attach %s and %s to a new node %s"
                                    % (left, right, entry["new_name"]))
                if round_index > 0:
                    frame["d_title"] = "D'"
                    frame["star_title"] = "D'*"
                specs.append(frame)
                step = step + 1
            built = growing

            # Recompute the matrix: one new entry at a time.
            for item in entry["replacements"]:
                frame = round_frame(entry, NEW_MS)
                frame["built"] = list(built)
                frame["totals_shown"] = set(shown_totals)
                frame["star_shown"] = set(full_star)
                frame["pair"] = entry["pair"]
                frame["cell_focus"] = set([(left, item["other"]),
                                           (item["other"], left),
                                           (right, item["other"]),
                                           (item["other"], right),
                                           (left, right), (right, left)])
                frame["lines"] = [
                    [("D'(%s,%s)  =  ( D(%s,%s) + D(%s,%s) - D(%s,%s) ) / 2"
                      % (entry["new_name"], item["other"], left, item["other"],
                         right, item["other"], left, right), DIM)],
                    [("D'(%s,%s)  =  ( %s + %s - %s ) / 2  =  "
                      % (entry["new_name"], item["other"], number(item["from_left"]),
                         number(item["from_right"]), number(item["across"])), INK),
                     (number(item["value"]), GREEN)],
                ]
                frame["caption"] = ("Every distance to the new node comes from the "
                                    "two it replaces")
                if round_index > 0:
                    frame["d_title"] = "D'"
                    frame["star_title"] = "D'*"
                specs.append(frame)
        else:
            # Base case: with three taxa every pair is a neighbor, so all three
            # limbs come straight out of the three-point formula.
            tie = round_frame(entry, HOLD_MS)
            tie["built"] = list(built)
            tie["totals_shown"] = set(shown_totals)
            tie["star_shown"] = set(full_star)
            tie["d_title"] = "D'"
            tie["star_title"] = "D'*"
            tie["caption"] = ("Every entry of D'* ties, so all three are neighbors: "
                              "hang them from one node")
            specs.append(tie)

            limb_lines = []
            for name in names:
                one, two = entry["sources"][name]
                limb_lines.append(
                    [("limb(%s)  =  ( %s + %s - %s ) / 2  =  "
                      % (name, number(entry["matrix"][(name, one)]),
                         number(entry["matrix"][(name, two)]),
                         number(entry["matrix"][(one, two)])), INK),
                     (number(entry["limbs"][name]), GREEN)])

            shown_lines = 1
            while shown_lines <= len(limb_lines):
                frame = round_frame(entry, LIMB_MS)
                frame["built"] = list(built)
                frame["totals_shown"] = set(shown_totals)
                frame["star_shown"] = set(full_star)
                frame["d_title"] = "D'"
                frame["star_title"] = "D'*"
                frame["lines"] = limb_lines[0:shown_lines]
                frame["caption"] = ("Each limb is half of its two distances minus "
                                    "the far one")
                specs.append(frame)
                shown_lines = shown_lines + 1

            growing = list(built)
            for edge in TREE_EDGES:
                if edge not in growing:
                    growing.append(edge)
            step = 1
            while step <= GROW_FRAMES:
                frame = round_frame(entry, GROW_MS)
                frame["built"] = list(growing)
                frame["totals_shown"] = set(shown_totals)
                frame["star_shown"] = set(full_star)
                frame["d_title"] = "D'"
                frame["star_title"] = "D'*"
                frame["growth"] = ease(step / GROW_FRAMES)
                frame["lines"] = limb_lines
                frame["caption"] = "Hang the last three limbs from a single node"
                specs.append(frame)
                step = step + 1
            built = growing
        round_index = round_index + 1

    # Bring the original D back for the last beat so the tree can be checked.
    first = ROUNDS[0]
    final = round_frame(first, FINAL_HOLD_MS)
    final["built"] = list(built)
    final["totals_shown"] = set(first["totals"].keys())
    for one, two in upper_pairs(first["names"]):
        final["star_shown"].add((one, two))
        final["star_shown"].add((two, one))
    final["caption"] = "The tree is additive: it reproduces every distance in D"
    specs.append(final)
    return specs


def draw_matrix(axis: "plt.Axes", left: float, title: str, names: "list[str]",
                values: dict, shown: set, totals: dict, totals_shown: set,
                row_focus: str, focus: set, pair: "tuple") -> None:
    """One labelled matrix. `shown` empty means every cell is drawn."""
    if len(names) == 0:
        return
    axis.text(left + 0.5 * CELL, MATRIX_TOP + 0.05, title, fontsize=TITLE_SIZE,
              color=INK, ha="center", va="center", fontproperties=OPTIMA_ITALIC,
              zorder=5)
    column = 0
    while column < len(names):
        if names[column] in pair:
            colour = RED
        elif names[column] == row_focus:
            colour = BLUE
        else:
            colour = INK
        axis.text(left + (column + 1.5) * CELL, MATRIX_TOP, names[column],
                  fontsize=HEADER_SIZE, color=colour, ha="center", va="center",
                  fontproperties=OPTIMA_ITALIC, zorder=5)
        axis.text(left + 0.5 * CELL, MATRIX_TOP - (column + 1) * CELL,
                  names[column], fontsize=HEADER_SIZE, color=colour, ha="center",
                  va="center", fontproperties=OPTIMA_ITALIC, zorder=5)
        column = column + 1

    if len(totals) > 0:
        axis.text(left + (len(names) + TOTAL_GAP) * CELL, MATRIX_TOP, "total",
                  fontsize=HEADER_SIZE - 1, color=DIM, ha="center", va="center",
                  fontproperties=OPTIMA_ITALIC, zorder=5)

    row = 0
    while row < len(names):
        column = 0
        while column < len(names):
            key = (names[row], names[column])
            visible = len(shown) == 0 or key in shown or row == column
            if visible:
                value = values[key]
                if key in focus:
                    colour = BLUE
                    font = MONO_BOLD
                elif (names[row] in pair and names[column] in pair
                      and names[row] != names[column]):
                    colour = RED
                    font = MONO_BOLD
                elif row == column:
                    colour = DIM
                    font = MONO
                elif names[row] == row_focus:
                    colour = BLUE
                    font = MONO
                else:
                    colour = INK
                    font = MONO
                text = "%g" % value
            else:
                colour = FAINT
                font = MONO
                text = "."
            axis.text(left + (column + 1.5) * CELL, MATRIX_TOP - (row + 1) * CELL,
                      text, fontsize=VALUE_SIZE, color=colour, ha="center",
                      va="center", fontproperties=font, zorder=5)
            column = column + 1
        if len(totals) > 0:
            if names[row] in totals_shown:
                if names[row] == row_focus:
                    colour = GREEN
                else:
                    colour = DIM
                text = "%g" % totals[names[row]]
            else:
                colour = FAINT
                text = "."
            axis.text(left + (len(names) + TOTAL_GAP) * CELL,
                      MATRIX_TOP - (row + 1) * CELL, text, fontsize=VALUE_SIZE,
                      color=colour, ha="center", va="center", fontproperties=MONO_BOLD,
                      zorder=5)
        row = row + 1


def draw_tree(axis: "plt.Axes", spec: dict) -> None:
    built = spec["built"]
    index = 0
    while index < len(built):
        edge = built[index]
        if spec["growth"] < 1.0 and index >= len(built) - 2 and len(built) == 2:
            reach = spec["growth"]
        elif spec["growth"] < 1.0 and index >= 2:
            reach = spec["growth"]
        else:
            reach = 1.0
        start = POSITIONS[edge[1]]
        end = POSITIONS[edge[0]]
        x = start[0] + (end[0] - start[0]) * reach
        y = start[1] + (end[1] - start[1]) * reach
        axis.plot([start[0], x], [start[1], y], color=INK, linewidth=1.9,
                  solid_capstyle="round", zorder=4)
        if reach > 0.98:
            label = LIMBS.get(edge[0], LIMBS.get(edge[1], 0))
            if edge == ("m", "c"):
                label = LIMBS["m"]
                axis.text((start[0] + end[0]) / 2, start[1] + 0.16, "%g" % label,
                          fontsize=LIMB_SIZE, color=INK, ha="center", va="bottom",
                          fontproperties=MONO, zorder=6)
            else:
                axis.text((start[0] + end[0]) / 2 + 0.14,
                          (start[1] + end[1]) / 2, "%g" % label,
                          fontsize=LIMB_SIZE, color=INK, ha="left", va="center",
                          fontproperties=MONO, zorder=6)
        index = index + 1

    drawn = set()
    for edge in built:
        drawn.add(edge[0])
        drawn.add(edge[1])
    for name in drawn:
        position = POSITIONS[name]
        if name in TAXA:
            disc = patches.Circle(position, LEAF_RADIUS, facecolor=DISC,
                                  edgecolor="none", zorder=5)
            axis.add_patch(disc)
            axis.text(position[0], position[1], name, fontsize=LEAF_SIZE,
                      color="white", ha="center", va="center",
                      fontproperties=OPTIMA_ITALIC, zorder=6)
        else:
            dot = patches.Circle(position, NODE_RADIUS, facecolor=INK,
                                 edgecolor="none", zorder=5)
            axis.add_patch(dot)
            if name == "m":
                axis.text(position[0] - 0.06, position[1] - 0.26, "m",
                          fontsize=LIMB_SIZE, color=INK, ha="center", va="top",
                          fontproperties=OPTIMA_ITALIC, zorder=6)


def draw_lines(axis: "plt.Axes", figure: "plt.Figure", lines: "list") -> None:
    """The working, one monospace line per row, each coloured piecewise."""
    if len(lines) == 0:
        return
    advance = mono_advance(figure, WORK_SIZE)
    index = 0
    while index < len(lines):
        pieces = lines[index]
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


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)
    draw_matrix(axis, D_LEFT, spec["d_title"], spec["names"], spec["values"],
                set(), spec["totals"], spec["totals_shown"], spec["row_focus"],
                spec["cell_focus"], spec["pair"])
    draw_matrix(axis, STAR_LEFT, spec["star_title"], spec["names"],
                spec["star_values"], spec["star_shown"], {}, set(), "",
                set([spec["star_focus"]]), spec["pair"])
    draw_tree(axis, spec)
    draw_lines(axis, figure, spec["lines"])
    if spec["caption"] != "":
        axis.text(FIGURE_WIDTH / 2, CAPTION_Y, spec["caption"], fontsize=LABEL_SIZE,
                  color=INK, ha="center", va="center", fontproperties=OPTIMA_ITALIC,
                  zorder=7)
    figure.savefig(output_path, transparent=True)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: example_neighbor_joining.py OUTPUT.gif")
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
    directory = tempfile.mkdtemp(prefix="neighbor_joining_")
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
