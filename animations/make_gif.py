"""Assemble rendered PNG frames into a looping GIF in a consistent way.

Usage from a frame-generator script:

    from make_gif import assemble_gif
    assemble_gif(frame_paths, "out.gif", width=800, height=600)

Convention: hold the first frame ~900 ms and the last ~1600 ms so viewers
can read the start and end states; ~130 ms per frame in between. Ease the
animated parameter with ease_in_out so motion starts and stops gently.
"""

import math

from PIL import Image


def ease_in_out(fraction: float) -> float:
    """Cosine ease: 0 -> 0, 1 -> 1, gentle at both ends."""
    return 0.5 * (1 - math.cos(math.pi * fraction))


def assemble_gif(
    frame_paths: "list[str]",
    output_path: str,
    width: int,
    height: int,
    middle_duration_ms: int = 130,
    first_hold_ms: int = 900,
    last_hold_ms: int = 1600,
    frame_durations: "list[int]" = None,
) -> None:
    """Resize frames and write a looping optimized GIF.

    Pass `frame_durations` (one value per frame, in ms) when the animation
    needs per-frame pacing, for example holding a frame long enough to read a
    caption that just changed. Identical consecutive frames collapse into a
    single frame with the summed delay, so long holds are nearly free in bytes.
    Without it, the first/middle/last convention applies.
    """
    images = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        image = image.resize((width, height), Image.LANCZOS)
        images.append(image)

    if frame_durations is not None:
        if len(frame_durations) != len(images):
            raise ValueError("frame_durations must have one value per frame")
        durations = list(frame_durations)
    else:
        durations = []
        frame_index = 0
        while frame_index < len(images):
            if frame_index == 0:
                durations.append(first_hold_ms)
            elif frame_index == len(images) - 1:
                durations.append(last_hold_ms)
            else:
                durations.append(middle_duration_ms)
            frame_index = frame_index + 1

    first_image = images[0]
    rest_images = images[1:]
    first_image.save(
        output_path,
        save_all=True,
        append_images=rest_images,
        duration=durations,
        loop=0,
        optimize=True,
    )
