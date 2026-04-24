"""Entry point for ``python -m gdb``; delegates to :mod:`gdb.cli`."""

from .cli import main

if __name__ == "__main__":
    main()
