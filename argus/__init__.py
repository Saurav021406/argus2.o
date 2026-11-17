"""Argus package.

NOTE: the old version of this file imported argus_server at import time, which
exits the process if MySQL is missing. That made every `import argus.<anything>`
pull in 442 MCP tools. The import is now lazy: only `argus.main()` touches it.
"""

__version__ = "1.1.0"


def main():
    """Entry point for the legacy MCP server (`argus` console script)."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from argus_server import main as _server_main

    return _server_main()


__all__ = ["main"]
