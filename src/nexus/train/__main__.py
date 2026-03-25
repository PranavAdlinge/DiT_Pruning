"""Entry point for `python -m nexus.train`. Avoids sys.modules warning from -m nexus.train.main."""

from .main import main

if __name__ == "__main__":
    main()
