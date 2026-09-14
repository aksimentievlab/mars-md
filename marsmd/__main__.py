"""``python -m marsmd`` -- alias for :func:`marsmd.run.main`."""

from __future__ import annotations

from .run import main

if __name__ == "__main__":
    raise SystemExit(main())
