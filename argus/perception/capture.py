"""Live screen capture that matches how the recorder saved training frames."""
from __future__ import annotations

import io

from PIL import Image

from argus.actions.platform import ensure_dpi_aware


class ScreenCapture:
    """grab() returns what the policy was trained on: downscaled to max_width, JPEG q85 round-trip."""

    def __init__(self, monitor: int = 1, max_width: int = 1280, jpeg_quality: int = 85):
        ensure_dpi_aware()
        import mss

        self._sct = mss.mss()
        self.monitor = self._sct.monitors[monitor]
        self.max_width, self.q = max_width, jpeg_quality
        self.screen_size = (self.monitor["width"], self.monitor["height"])

    def grab(self) -> Image.Image:
        shot = self._sct.grab(self.monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        scale = min(1.0, self.max_width / img.width)
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.q)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
