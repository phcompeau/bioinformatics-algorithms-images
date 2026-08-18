"""Animated greedy motif search, in the 02-180 lecture style.

Uses the three strings from Hidden_Messages.pptx slide 160 with k = 3. For each
k-mer of the first string, build a profile from the motifs chosen so far, take the
profile-most-probable k-mer in the next string, and keep the collection with the
lowest score. Pseudocounts are included, as the deck insists.

Asserted against a brute-force search over every combination of one k-mer per
string: the greedy answer's score is reported honestly alongside the true optimum,
since greedy is a heuristic and need not find it.

Run:  python3 greedy_motif_search.py OUTPUT.gif
"""

import os
import sys
import tempfile

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from lecture_style import (BACKGROUND, DIM, FAINT, GREEN, INK, MONO, MONO_BOLD,
                           OPTIMA, OPTIMA_ITALIC, RED, new_axes)
from make_gif import assemble_gif

DNA = ["TACAGAC", "ACCCAGT", "CAGCATT"]
K = 3
ALPHABET = "ACGT"

FIGURE_WIDTH = 9.6
FIGURE_HEIGHT = 4.0
RENDER_DPI = 145
OUTPUT_WIDTH = 1392
OUTPUT_HEIGHT = 580

CHAR_ADVANCE = 0.30
CHAR_SIZE = 22.0
DNA_LEFT = 0.75
DNA_TOP = 3.25
DNA_GAP = 0.62

PROFILE_LEFT = 5.35
PROFILE_TOP = 3.25
PROFILE_CELL_X = 0.78
PROFILE_CELL_Y = 0.52
PROFILE_SIZE = 13.0

SCORE_Y = 1.02
SCORE_SIZE = 24.0

STEP_MS = 1500
PICK_MS = 1300
BEST_MS = 2000
FIRST_HOLD_MS = 2600
FINAL_HOLD_MS = 5200


def kmers(text: str) -> "list[str]":
    result = []
    start = 0
    while start + K <= len(text):
        result.append(text[start:start + K])
        start = start + 1
    return result


def profile_of(motifs: "list[str]") -> "list[dict]":
    """Column-wise letter frequencies with Laplace pseudocounts."""
    columns = []
    column = 0
    while column < K:
        counts = {}
        for letter in ALPHABET:
            counts[letter] = 1
        for motif in motifs:
            counts[motif[column]] = counts[motif[column]] + 1
        total = 0
        for letter in ALPHABET:
            total = total + counts[letter]
        frequencies = {}
        for letter in ALPHABET:
            frequencies[letter] = counts[letter] / total
        columns.append(frequencies)
        column = column + 1
    return columns


def probability(text: str, profile: "list[dict]") -> float:
    result = 1.0
    column = 0
    while column < K:
        result = result * profile[column][text[column]]
        column = column + 1
    return result


def most_probable(text: str, profile: "list[dict]") -> "tuple[int, str]":
    best_index = 0
    best_value = -1.0
    candidates = kmers(text)
    index = 0
    while index < len(candidates):
        value = probability(candidates[index], profile)
        if value > best_value + 1e-12:
            best_value = value
            best_index = index
        index = index + 1
    return (best_index, candidates[best_index])


def score_of(motifs: "list[str]") -> int:
    """Total mismatches against the consensus, column by column."""
    total = 0
    column = 0
    while column < K:
        counts = {}
        for letter in ALPHABET:
            counts[letter] = 0
        for motif in motifs:
            counts[motif[column]] = counts[motif[column]] + 1
        best = 0
        for letter in ALPHABET:
            if counts[letter] > best:
                best = counts[letter]
        total = total + len(motifs) - best
        column = column + 1
    return total


def greedy() -> "list[dict]":
    """Record each candidate start in the first string and what it produced."""
    record = []
    best_motifs = []
    for text in DNA:
        best_motifs.append(text[0:K])
    best_score = score_of(best_motifs)
    starts = kmers(DNA[0])
    index = 0
    while index < len(starts):
        motifs = [starts[index]]
        picks = [index]
        stages = []
        row = 1
        while row < len(DNA):
            profile = profile_of(motifs)
            chosen_index, chosen = most_probable(DNA[row], profile)
            stages.append({"row": row, "profile": profile, "index": chosen_index,
                           "kmer": chosen, "motifs": list(motifs)})
            motifs.append(chosen)
            picks.append(chosen_index)
            row = row + 1
        current = score_of(motifs)
        improved = current < best_score
        if improved:
            best_motifs = list(motifs)
            best_score = current
        record.append({"start": index, "motifs": list(motifs), "picks": list(picks),
                       "score": current, "stages": stages, "improved": improved,
                       "best_score": best_score, "best_motifs": list(best_motifs)})
        index = index + 1
    return record


ROUNDS = greedy()
BEST_MOTIFS = ROUNDS[len(ROUNDS) - 1]["best_motifs"]
BEST_SCORE = ROUNDS[len(ROUNDS) - 1]["best_score"]


def brute_force() -> "tuple[int, list]":
    best = None
    best_choice = []
    first = kmers(DNA[0])
    second = kmers(DNA[1])
    third = kmers(DNA[2])
    for a in first:
        for b in second:
            for c in third:
                value = score_of([a, b, c])
                if best is None or value < best:
                    best = value
                    best_choice = [a, b, c]
    return (best, best_choice)


OPTIMUM, OPTIMUM_MOTIFS = brute_force()


def verify() -> "list[str]":
    lines = []

    for text in DNA:
        assert len(text) == len(DNA[0]), "strings must be equal length here"
        for letter in text:
            assert letter in ALPHABET, "unexpected symbol %s" % letter
    assert len(ROUNDS) == len(DNA[0]) - K + 1, "one round per k-mer of the first string"
    lines.append("%d strings of length %d, k = %d, so %d rounds"
                 % (len(DNA), len(DNA[0]), K, len(ROUNDS)))

    for entry in ROUNDS:
        assert len(entry["motifs"]) == len(DNA), "one motif per string"
        row = 0
        while row < len(DNA):
            start = entry["picks"][row]
            assert DNA[row][start:start + K] == entry["motifs"][row], (
                "motif %d does not come from its string" % row)
            row = row + 1
        assert entry["score"] == score_of(entry["motifs"]), "score mismatch"
    lines.append("every chosen motif really is a k-mer at its recorded position")

    for entry in ROUNDS:
        for stage in entry["stages"]:
            profile = stage["profile"]
            column = 0
            while column < K:
                total = 0.0
                for letter in ALPHABET:
                    assert profile[column][letter] > 0, (
                        "pseudocounts should keep every probability positive")
                    total = total + profile[column][letter]
                assert abs(total - 1.0) < 1e-9, "profile column must sum to 1"
                column = column + 1
            best = probability(stage["kmer"], profile)
            for candidate in kmers(DNA[stage["row"]]):
                assert probability(candidate, profile) <= best + 1e-12, (
                    "a more probable k-mer than %s exists" % stage["kmer"])
    lines.append("every profile is a valid distribution and each pick is the most probable")

    running = None
    for entry in ROUNDS:
        if running is None or entry["best_score"] < running:
            running = entry["best_score"]
        assert entry["best_score"] == running, "the best score must never worsen"
    lines.append("the best score never worsens across rounds, ending at %d" % BEST_SCORE)

    assert BEST_SCORE >= OPTIMUM, "greedy cannot beat the brute-force optimum"
    lines.append("greedy scores %d; brute force over all %d combinations gives %d"
                 % (BEST_SCORE, (len(DNA[0]) - K + 1) ** len(DNA), OPTIMUM))

    span = len(DNA[0]) * CHAR_ADVANCE
    assert DNA_LEFT + span < PROFILE_LEFT - 0.25, "strings collide with the profile"
    assert PROFILE_LEFT + (K + 1) * PROFILE_CELL_X < FIGURE_WIDTH - 0.2, (
        "profile runs off the right")
    lowest = DNA_TOP - (len(DNA) - 1) * DNA_GAP
    assert lowest - 0.3 > SCORE_Y + 0.3, "strings collide with the score"
    lines.append("strings span %.2f in, profile starts at %.2f in"
                 % (span, PROFILE_LEFT))
    return lines


def base_frame(duration: int) -> dict:
    return {"highlight": {}, "profile": [], "score": -1, "best": -1,
            "flash": False, "duration_ms": duration}


def build_specs() -> "list[dict]":
    specs = []
    opening = base_frame(FIRST_HOLD_MS)
    specs.append(opening)

    for entry in ROUNDS:
        chosen = {0: entry["picks"][0]}
        first = base_frame(STEP_MS)
        first["highlight"] = dict(chosen)
        first["best"] = entry["best_score"]
        specs.append(first)

        for stage in entry["stages"]:
            show = base_frame(PICK_MS)
            chosen = dict(chosen)
            chosen[stage["row"]] = stage["index"]
            show["highlight"] = dict(chosen)
            show["profile"] = stage["profile"]
            show["best"] = entry["best_score"]
            specs.append(show)

        done = base_frame(BEST_MS)
        done["highlight"] = dict(chosen)
        done["profile"] = profile_of(entry["motifs"])
        done["score"] = entry["score"]
        done["best"] = entry["best_score"]
        done["flash"] = entry["improved"]
        specs.append(done)

    final = base_frame(FINAL_HOLD_MS)
    picks = {}
    row = 0
    while row < len(DNA):
        picks[row] = DNA[row].index(BEST_MOTIFS[row])
        row = row + 1
    final["highlight"] = dict(picks)
    final["profile"] = profile_of(BEST_MOTIFS)
    final["score"] = BEST_SCORE
    final["best"] = BEST_SCORE
    specs.append(final)
    return specs


def draw_frame(spec: dict, output_path: str) -> None:
    figure, axis = new_axes(FIGURE_WIDTH, FIGURE_HEIGHT, RENDER_DPI)

    row = 0
    while row < len(DNA):
        y = DNA_TOP - row * DNA_GAP
        start = spec["highlight"].get(row, -1)
        if start >= 0:
            box = patches.Rectangle(
                (DNA_LEFT + start * CHAR_ADVANCE - 0.04, y - 0.25),
                K * CHAR_ADVANCE + 0.08, 0.5, facecolor="#E2DCCB",
                edgecolor=GREEN, linewidth=1.6, zorder=3)
            axis.add_patch(box)
        column = 0
        while column < len(DNA[row]):
            if start >= 0 and start <= column < start + K:
                colour = INK
            else:
                colour = DIM
            axis.text(DNA_LEFT + (column + 0.5) * CHAR_ADVANCE, y, DNA[row][column],
                      fontsize=CHAR_SIZE, color=colour, ha="center", va="center",
                      fontproperties=MONO_BOLD, zorder=5)
            column = column + 1
        row = row + 1

    if len(spec["profile"]) > 0:
        index = 0
        while index < len(ALPHABET):
            letter = ALPHABET[index]
            y = PROFILE_TOP - index * PROFILE_CELL_Y
            axis.text(PROFILE_LEFT, y, letter, fontsize=PROFILE_SIZE, color=INK,
                      ha="center", va="center", fontproperties=MONO_BOLD, zorder=5)
            column = 0
            while column < K:
                value = spec["profile"][column][letter]
                axis.text(PROFILE_LEFT + (column + 1) * PROFILE_CELL_X, y,
                          "%.2f" % value, fontsize=PROFILE_SIZE, color=DIM,
                          ha="center", va="center", fontproperties=MONO, zorder=5)
                column = column + 1
            index = index + 1

    if spec["score"] >= 0:
        if spec["flash"]:
            colour = GREEN
        else:
            colour = DIM
        axis.text(FIGURE_WIDTH / 2 - 0.6, SCORE_Y, str(spec["score"]),
                  fontsize=SCORE_SIZE, color=colour, ha="center", va="center",
                  fontproperties=MONO_BOLD, zorder=7)
    if spec["best"] >= 0:
        axis.text(FIGURE_WIDTH / 2 + 0.6, SCORE_Y, str(spec["best"]),
                  fontsize=SCORE_SIZE, color=INK, ha="center", va="center",
                  fontproperties=MONO_BOLD, zorder=7)

    figure.savefig(output_path, facecolor=BACKGROUND)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: greedy_motif_search.py OUTPUT.gif")
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
    directory = tempfile.mkdtemp(prefix="motif_")
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
