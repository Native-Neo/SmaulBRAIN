"""Entry point: python -m smaulbrain <train|infer|report|quantize> [flags]."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
