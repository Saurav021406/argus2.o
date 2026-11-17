"""Platform helpers."""
import sys


def ensure_dpi_aware() -> None:
    """Make mss, pynput and pyautogui agree on pixel coordinates.

    On Windows with display scaling (125%/150%) these libraries can report
    different coordinate systems unless the process is DPI-aware. Call this
    once at start-up, before any capture or input library is used.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
