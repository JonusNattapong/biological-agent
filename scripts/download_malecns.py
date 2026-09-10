"""Download the core official MaleCNS v1.0 bulk tables.

The large connectivity file is opt-in via ``--weights`` so a normal setup does
not unexpectedly download ~1.1 GB.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen

from neurofly.brain.malecns import (
    ANNOTATIONS_FILENAME,
    NEUROTRANSMITTERS_FILENAME,
    WEIGHTS_FILENAME,
    official_bulk_urls,
)


def download(url: str, destination: Path, *, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"skip  {destination} (already exists)")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "NeuroFly-MaleCNS/0.1"})
    with urlopen(request, timeout=60) as response, partial.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0") or 0)
        received = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            received += len(chunk)
            if total:
                pct = received * 100.0 / total
                print(
                    f"\rget   {destination.name}: {received / 2**20:.1f}/{total / 2**20:.1f} MiB ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
        if total:
            print()

    partial.replace(destination)
    print(f"saved {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official MaleCNS v1.0 bulk data")
    parser.add_argument("--weights", action="store_true", help="also download the ~1.1 GB full connection graph")
    parser.add_argument("--force", action="store_true", help="replace existing files")
    parser.add_argument("--output", default="data/raw", help="destination directory")
    args = parser.parse_args()

    out = Path(args.output)
    urls = official_bulk_urls()
    files = [
        (urls["annotations"], out / ANNOTATIONS_FILENAME),
        (urls["neurotransmitters"], out / NEUROTRANSMITTERS_FILENAME),
    ]
    if args.weights:
        files.append((urls["weights"], out / WEIGHTS_FILENAME))

    for url, path in files:
        download(url, path, force=args.force)


if __name__ == "__main__":
    main()
