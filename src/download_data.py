# To download MC:  python src/download_data.py --prefix MediumEnergy_FHC_StandardMC_Playlist
# To download data: python src/download_data.py --prefix MediumEnergy_FHC_Data_Playlist

import argparse
import os
import string
import subprocess
import sys
import urllib.request

# 1A to 1P, each file contains a list of ROOT files. Download them using xrdcp
# into /pscratch/sd/g/gregork/MINERvA/raw_data (create dir if not exists).
# Example playlist:
# https://minerva.fnal.gov/wp-content/uploads/2025/10/MediumEnergy_FHC_StandardMC_Playlist1A.txt


DEFAULT_BASE_URL = "https://minerva.fnal.gov/wp-content/uploads/2025/10"
DEFAULT_PREFIX = "MediumEnergy_FHC_StandardMC_Playlist"
scratch_dir = os.environ["SCRATCH"]


def default_playlists():
    #return [f"1{letter}" for letter in string.ascii_uppercase[:16]]
    # return above but without H,I,J,K
    return ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "1L", "1M", "1N", "1O", "1P"]


def build_playlist_url(base_url, prefix, playlist_id):
    return f"{base_url}/{prefix}{playlist_id}.txt"


def read_playlist(url):
    with urllib.request.urlopen(url) as response:
        content = response.read().decode("utf-8")
    entries = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def run_xrdcp(src, dest, force=False, dry_run=False):
    if os.path.exists(dest) and not force:
        print(f"skip: {dest} already exists")
        return 0
    cmd = ["xrdcp"]
    if force:
        cmd.append("-f")
    cmd.extend([src, dest])
    if dry_run:
        print(f"dry-run: {' '.join(cmd)}")
        return 0
    print(f"download: {src} -> {dest}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Download MINERvA ROOT files via xrdcp.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for playlist text files.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Playlist filename prefix.",
    )
    parser.add_argument(
        "--playlists",
        nargs="*",
        default=None,
        help="Playlist IDs to download (default: 1A-1P).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print xrdcp commands without executing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only download the first N files of each playlist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    playlists = args.playlists if args.playlists else default_playlists()
    DEFAULT_OUTPUT_DIR = os.path.join(scratch_dir, "MINERvA/raw_data", args.prefix)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    failures = 0
    for playlist_id in playlists:
        playlist_url = build_playlist_url(args.base_url, args.prefix, playlist_id)
        print(f"playlist: {playlist_id} ({playlist_url})")
        try:
            entries = read_playlist(playlist_url)
        except Exception as exc:
            print(f"error: failed to read playlist {playlist_id}: {exc}", file=sys.stderr)
            failures += 1
            continue

        if args.limit is not None:
            entries = entries[: args.limit]

        playlist_dir = os.path.join(DEFAULT_OUTPUT_DIR, playlist_id)
        os.makedirs(playlist_dir, exist_ok=True)

        for entry in entries:
            filename = os.path.basename(entry)
            dest = os.path.join(playlist_dir, filename)
            rc = run_xrdcp(entry, dest, force=args.force, dry_run=args.dry_run)
            if rc != 0:
                failures += 1

    if failures:
        print(f"completed with {failures} failures", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

