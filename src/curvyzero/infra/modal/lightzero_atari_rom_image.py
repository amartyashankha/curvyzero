"""Modal image helper for stock LightZero Atari environments.

Atari/ALE ROMs are not bundled by Gym, Gymnasium, LightZero, or ale-py. This
helper is intentionally separate so the AutoROM license acceptance is visible
at the image boundary, before any stock Atari Pong environment or trainer runs.
"""

from __future__ import annotations

import modal

OPENCV_PYTHON_HEADLESS_VERSION = "4.11.0.86"
AUTOROM_VERSION = "0.6.1"
ATARI_ROM_LICENSE_NOTICE = (
    "This Modal image installs AutoROM[accept-rom-license] and runs "
    "`AutoROM --accept-license` during image build so ALE can load Atari ROMs."
)


def build_lightzero_atari_rom_image(*, lightzero_version: str) -> modal.Image:
    """Build the smallest LightZero Atari image with explicit ROM handling."""

    return (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            f"LightZero=={lightzero_version}",
            f"opencv-python-headless=={OPENCV_PYTHON_HEADLESS_VERSION}",
            f"AutoROM[accept-rom-license]=={AUTOROM_VERSION}",
        )
        .run_commands("AutoROM --accept-license")
    )
