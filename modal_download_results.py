"""
modal_download_results.py
════════════════════════════════════════════════════════════════════════════════
Download all benchmark outputs from the Modal persistent volume to your Mac.

Usage:
  python modal_download_results.py
  python modal_download_results.py --dest ./my_results
════════════════════════════════════════════════════════════════════════════════
"""

import modal
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="./modal_results",
                        help="Local directory to download results into")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    output_vol = modal.Volume.from_name("vags-outputs")

    print(f"Downloading results from Modal volume 'vags-outputs' → {dest}")
    count = 0
    for entry in output_vol.listdir("/", recursive=True):
        remote_path = entry.path
        local_path  = dest / remote_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in output_vol.read_file(remote_path):
                f.write(chunk)
        print(f"  ✓ {remote_path}")
        count += 1

    print(f"\nDone. {count} files downloaded to {dest.resolve()}")


if __name__ == "__main__":
    main()
