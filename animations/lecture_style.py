"""Shared visual style for animations that follow Phillip's 02-180 lecture decks.

Measured from Read_Mapping.pptx: cream ground, white nodes with a thin dark
outline, no arrowheads, large monospace letters, dark discs for indices. The
decks set their monospace in Consolas, which is not installed on this Mac, so
Menlo stands in as the closest match.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

BACKGROUND = "#EEE9DF"
INK = "#1A1A1A"
DIM = "#9A958C"
FAINT = "#CFC9BC"
BLUE = "#176FC1"
GREEN = "#149B52"
RED = "#ED1C24"
PURPLE = "#95319E"
ORANGE = "#F46E2B"
DISC = "#3F3F3F"

MONO = font_manager.FontProperties(family="Menlo")
MONO_BOLD = font_manager.FontProperties(family="Menlo", weight="bold")
OPTIMA = font_manager.FontProperties(family="Optima")
OPTIMA_ITALIC = font_manager.FontProperties(family="Optima", style="italic")


def new_axes(width: float, height: float, dpi: int) -> "tuple":
    """Cream axes filling the whole figure, so no resolution is lost to margins."""
    figure = plt.figure(figsize=(width, height), dpi=dpi)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect("equal")
    axis.axis("off")
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    return (figure, axis)


def ease(fraction: float) -> float:
    """Cosine ease: gentle at both ends."""
    clamped = min(max(fraction, 0.0), 1.0)
    return 0.5 * (1 - math.cos(math.pi * clamped))


def staggered_progress(index: int, count: int, step: float, stagger: float) -> float:
    """Eased progress for one item in a wave, so rows do not all move at once."""
    if count <= 1:
        return ease(step)
    delay = stagger * (index / (count - 1.0))
    local = (step - delay) / (1.0 - stagger)
    return ease(min(max(local, 0.0), 1.0))


def transit_alpha(progress: float, moving: bool) -> float:
    """Fade an element while it travels; at full opacity crossings smear."""
    if not moving:
        return 1.0
    return 1.0 - 0.78 * math.sin(math.pi * min(max(progress, 0.0), 1.0))


def sort_permutation(items: "list[str]") -> "list[int]":
    """Indices of items in lexicographic order."""
    pairs = []
    index = 0
    while index < len(items):
        pairs.append((items[index], index))
        index = index + 1
    pairs.sort()
    order = []
    for value, index in pairs:
        order.append(index)
    return order
