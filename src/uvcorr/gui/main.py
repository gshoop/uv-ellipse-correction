"""Entry point for the ``uvcorr-gui`` console script.

The main window (plan section 9) arrives in phase 4; until then the command
only reports that the GUI is not implemented yet.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``uvcorr-gui [file]``. Returns a process exit code."""
    del argv  # the optional file argument is handled once the GUI exists
    print("uvcorr-gui: not implemented yet (planned for phase 4).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
