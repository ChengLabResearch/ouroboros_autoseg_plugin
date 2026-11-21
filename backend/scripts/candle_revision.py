"""Read the resolved Candle SAM3 revision and optionally verify backend logs."""

import argparse
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        raise SystemExit("Python 3.10 requires tomli: python3 -m pip install tomli")


def resolved_revision(lockfile):
    with lockfile.open("rb") as handle:
        packages = tomllib.load(handle).get("package", [])
    matches = [package for package in packages if package.get("name") == "candle-sam3"]
    if len(matches) != 1:
        raise ValueError("Cargo.lock must contain exactly one candle-sam3 package")
    source = matches[0].get("source", "")
    revision = re.fullmatch(r"git\+[^#]+#([0-9a-fA-F]{40})", source)
    if revision is None:
        raise ValueError("candle-sam3 must have a Git source with a resolved 40-character SHA")
    return revision.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lockfile", type=Path)
    parser.add_argument("logfile", type=Path, nargs="?")
    args = parser.parse_args()
    try:
        expected = resolved_revision(args.lockfile)
        if args.logfile is not None:
            observed = re.findall(r"\bcandle_sha=(\S*)", args.logfile.read_text())
            if not observed:
                raise ValueError("backend log contains no candle_sha field")
            if any(revision != expected for revision in observed):
                raise ValueError(f"expected candle_sha={expected}; observed {sorted(set(observed))}")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(expected)


if __name__ == "__main__":
    main()
