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


def assemble_transparent_gif(
    frame_paths: "list[str]",
    output_path: str,
    width: int,
    height: int,
    frame_durations: "list[int]",
    matte: str = "#EEE9DF",
    colours: int = 64,
) -> None:
    """Write a looping GIF whose background is transparent.

    The frames must be RGBA (saved with `transparent=True`), because GIF
    transparency is a single palette index rather than an alpha channel: the only
    safe way to tell "background" from "white ink" is the alpha channel itself.
    White leaf numbers on dark discs would otherwise be punched into holes.

    Anti-aliased edge pixels cannot be half transparent in a GIF, so anything even
    slightly opaque is composited against `matte` and kept. That covers deliberate
    fades too: a row at 20% opacity mid-flight would otherwise blink out entirely
    rather than fading. Pick the colour the page will actually use and those pixels
    are exact; on a different background they carry a faint fringe of the matte.
    """
    matte_rgb = (int(matte[1:3], 16), int(matte[3:5], 16), int(matte[5:7], 16))
    # Transparent frames cannot be diffed against each other, so each one is
    # stored whole. A small palette is what keeps that affordable: these figures
    # use few real colours, and quantizing to hundreds of near-duplicates only
    # adds noise for the compressor to store.
    clear_index = colours
    frames = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGBA")
        image = image.resize((width, height), Image.LANCZOS)
        alpha = image.getchannel("A")
        backdrop = Image.new("RGB", image.size, matte_rgb)
        backdrop.paste(image.convert("RGB"), (0, 0), alpha)
        # One palette entry past the ink is reserved for transparency.
        flat = backdrop.convert("P", palette=Image.ADAPTIVE, colors=colours)
        clear = alpha.point(lambda value: 255 if value < 8 else 0)
        flat.paste(clear_index, (0, 0), clear)
        frames.append(flat)

    if len(frame_durations) != len(frames):
        raise ValueError("frame_durations must have one value per frame")

    first_frame = frames[0]
    first_frame.save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=list(frame_durations),
        loop=0,
        transparency=clear_index,
        disposal=2,
        optimize=False,
    )
