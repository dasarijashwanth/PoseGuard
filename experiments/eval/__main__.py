"""Allow `python -m experiments.eval` as a shortcut for the CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
