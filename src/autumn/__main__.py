"""Allows `python -m autumn` to run the CLI entry point."""

from autumn.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
