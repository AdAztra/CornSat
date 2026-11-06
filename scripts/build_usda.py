"""Build USDA NASS processed tables. Exit code 2 if NASS_API_KEY is missing."""
import sys

from cornsat.usda import NassKeyMissing, build_usda


def main() -> int:
    try:
        out = build_usda()
    except NassKeyMissing as e:
        print(e)
        return 2
    for name, df in out.items():
        print(f"{name}: {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
