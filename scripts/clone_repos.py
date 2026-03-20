"""Clone all repositories referenced in the CSV dataset files and pre-fetch
only the blobs needed for the commits in the dataset.

Two-phase approach:
  Phase 1: Blobless clone (--filter=blob:none --no-checkout) — fast, parallel
  Phase 2: Pre-fetch blobs for dataset commits via git cat-file --batch

Usage:
    python scripts/clone_repos.py \
        --csv datasets/dataset2-mr-advisory-cpp-groupstrat-seed17-train.csv \
        --csv datasets/dataset2-mr-advisory-cpp-groupstrat-seed17-val.csv \
        --csv datasets/dataset2-mr-advisory-cpp-groupstrat-seed17-test.csv \
        --output_dir /path/to/commits \
        --workers 8
"""

import argparse
import csv
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from urllib.parse import urlparse


print_lock = Lock()
NULL_OID = "0" * 40
FULL_CLONE_THRESHOLD = 200  # repos with >= this many commits get a full clone

# Prevent git from ever prompting for credentials (blocks password prompts).
GIT_ENV = {
    **os.environ,
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "echo",
    "GIT_SSH_COMMAND": "ssh -o BatchMode=yes",
}


def url_to_repo_name(project_url):
    """Convert a project URL to a repo name (path after domain)."""
    parsed = urlparse(project_url)
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path


def url_to_dir_name(project_url):
    """Convert a project URL to the directory name used by the pipeline."""
    return url_to_repo_name(project_url).replace("/", "_")


def collect_repos_and_commits(csv_paths):
    """Collect unique URLs and group commit hashes by repo URL."""
    repo_commits = defaultdict(set)
    for csv_path in csv_paths:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                repo_commits[row["project_url"]].add(row["commit_id"])
    return repo_commits


# ---------------------------------------------------------------------------
# Phase 1: blobless clone
# ---------------------------------------------------------------------------

def clone_repo(project_url, output_dir, full_clone=False):
    """Clone a single repo. Returns (url, success, message)."""
    dir_name = url_to_dir_name(project_url)
    dest = os.path.join(output_dir, dir_name)

    if os.path.exists(dest):
        return (project_url, True, "skipped (exists)")

    clone_url = project_url
    if not clone_url.endswith(".git"):
        clone_url += ".git"

    cmd = ["git", "clone", "--quiet", clone_url, dest]
    if not full_clone:
        cmd.insert(2, "--filter=blob:none")
        cmd.insert(3, "--no-checkout")

    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=600, env=GIT_ENV,
        )
        mode = "full" if full_clone else "blobless"
        return (project_url, True, f"cloned ({mode})")
    except subprocess.CalledProcessError as e:
        return (project_url, False, e.stderr.strip())
    except subprocess.TimeoutExpired:
        return (project_url, False, "timed out after 600s")


def run_clone_phase(repo_commits, output_dir, workers):
    """Phase 1: clone all repos in parallel (full for heavy repos, blobless for rest)."""
    urls = sorted(repo_commits.keys())
    total = len(urls)
    n_full = sum(1 for u in urls if len(repo_commits[u]) >= FULL_CLONE_THRESHOLD)
    print(f"Phase 1: Cloning {total} repos ({n_full} full, {total - n_full} blobless) with {workers} workers\n")

    success = 0
    failed_urls = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(clone_repo, url, output_dir,
                        full_clone=len(repo_commits[url]) >= FULL_CLONE_THRESHOLD): url
            for url in urls
        }
        for i, future in enumerate(as_completed(futures), 1):
            url, ok, msg = future.result()
            with print_lock:
                status = "OK" if ok else "FAIL"
                print(f"  [{i}/{total}] {status}: {url_to_dir_name(url)} — {msg}")
            if not ok:
                failed_urls.append(url)
            else:
                success += 1

    print(f"\nPhase 1 done. Cloned: {success}, Failed: {len(failed_urls)}")
    return failed_urls


# ---------------------------------------------------------------------------
# Phase 2: pre-fetch blobs for dataset commits
# ---------------------------------------------------------------------------

def prefetch_blobs(project_url, commit_hashes, output_dir):
    """Enumerate blob OIDs via diff-tree, then batch-fetch via cat-file.

    Returns (url, num_blobs_fetched, message).
    """
    dir_name = url_to_dir_name(project_url)
    repo_path = os.path.join(output_dir, dir_name)

    if not os.path.exists(repo_path):
        return (project_url, 0, "repo missing, skipped")

    # skip repos that were fully cloned (all blobs already local)
    result = subprocess.run(
        ["git", "-C", repo_path, "config", "--get", "remote.origin.promisor"],
        capture_output=True, text=True,
    )
    if result.stdout.strip() != "true":
        return (project_url, 0, "full clone, skipped prefetch")

    hashes = list(commit_hashes)
    if not hashes:
        return (project_url, 0, "no commits")

    # diff-tree --stdin: reads commit hashes, outputs changed blob OIDs
    # Trees are local (fetched during blobless clone), so this is fast.
    try:
        input_data = "\n".join(hashes) + "\n"
        result = subprocess.run(
            ["git", "-C", repo_path, "diff-tree", "-r", "--stdin", "--no-commit-id"],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=300,
            env=GIT_ENV,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        return (project_url, 0, f"diff-tree failed: {e}")

    blob_oids = set()
    for line in result.stdout.splitlines():
        if not line.startswith(":"):
            continue
        parts = line.split("\t")[0].split()
        if len(parts) >= 5:
            old_oid, new_oid = parts[2], parts[3]
            if old_oid != NULL_OID:
                blob_oids.add(old_oid)
            if new_oid != NULL_OID:
                blob_oids.add(new_oid)

    if not blob_oids:
        return (project_url, 0, "no blobs needed")

    # Bulk-fetch missing blobs via `git fetch origin` with explicit OIDs.
    # This does a single pack negotiation instead of cat-file's serial
    # one-object-at-a-time lazy fetching, which is much faster and avoids
    # timeouts on large repos.
    try:
        oid_list = list(blob_oids)
        BATCH = 500
        for start in range(0, len(oid_list), BATCH):
            batch = oid_list[start : start + BATCH]
            subprocess.run(
                ["git", "-C", repo_path, "fetch", "origin", *batch],
                capture_output=True,
                timeout=300,
                env=GIT_ENV,
            )
    except subprocess.TimeoutExpired:
        return (project_url, 0, "fetch timed out")
    except subprocess.CalledProcessError as e:
        # Fallback: try cat-file for whatever remains unfetched
        pass

    # Verify objects are present; count how many we actually have.
    try:
        oid_input = "\n".join(blob_oids) + "\n"
        proc = subprocess.Popen(
            ["git", "-C", repo_path, "cat-file", "--batch-check"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=GIT_ENV,
        )
        stdout, _ = proc.communicate(input=oid_input.encode(), timeout=60)
        fetched = sum(1 for line in stdout.decode().splitlines() if "missing" not in line)
    except (subprocess.TimeoutExpired, OSError):
        if proc.poll() is None:
            proc.kill()
        fetched = len(blob_oids)  # assume success if verify fails

    return (project_url, fetched, f"fetched {fetched} blobs")


def run_prefetch_phase(repo_commits, output_dir, workers):
    """Phase 2: pre-fetch blobs for dataset commits in parallel."""
    repos = sorted(repo_commits.keys())
    total = len(repos)
    print(f"\nPhase 2: Pre-fetching blobs for {total} repos with {workers} workers\n")

    total_blobs = 0
    failed = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(prefetch_blobs, url, repo_commits[url], output_dir): url
            for url in repos
        }
        for i, future in enumerate(as_completed(futures), 1):
            url, num_blobs, msg = future.result()
            total_blobs += num_blobs
            with print_lock:
                print(f"  [{i}/{total}] {url_to_dir_name(url)} — {msg}")
            if num_blobs == 0 and "fetched" not in msg and "no " not in msg:
                failed.append(url)

    print(f"\nPhase 2 done. Total blobs fetched: {total_blobs}")
    return failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clone repos and pre-fetch blobs for dataset commits"
    )
    parser.add_argument(
        "--csv", action="append", required=True,
        help="CSV file(s) to read (can specify multiple times)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to clone repos into",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel threads (default: 8)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    repo_commits = collect_repos_and_commits(args.csv)
    total_commits = sum(len(v) for v in repo_commits.values())
    print(f"Found {len(repo_commits)} repos, {total_commits} commits total.\n")

    # Phase 1: blobless clone
    clone_failures = run_clone_phase(repo_commits, args.output_dir, args.workers)

    # Phase 2: pre-fetch only the blobs we need
    prefetch_failures = run_prefetch_phase(repo_commits, args.output_dir, args.workers)

    if clone_failures:
        print(f"\nFailed clones ({len(clone_failures)}):")
        for url in clone_failures:
            print(f"  {url}")

    if prefetch_failures:
        print(f"\nFailed prefetches ({len(prefetch_failures)}):")
        for url in prefetch_failures:
            print(f"  {url}")


if __name__ == "__main__":
    main()
