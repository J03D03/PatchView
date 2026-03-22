"""Clone all repositories referenced in the CSV dataset files and pre-fetch
only the blobs needed for the commits in the dataset.

Two-phase approach (aligned with RepoSPD's data_loader.py):
  Phase 1: Clone (full for heavy repos, blobless for the rest)
  Phase 2: Pre-fetch blob OIDs for blobless repos via diff-tree + git fetch

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
import errno
import os
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from urllib.parse import urlparse


print_lock = Lock()
NULL_OID = "0" * 40
FULL_CLONE_THRESHOLD = 200

GIT_ENV = {
    **os.environ,
    "GIT_TERMINAL_PROMPT": "0",
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
# Phase 1: clone
# ---------------------------------------------------------------------------

def clone_repo(project_url, output_dir, full_clone=False):
    """Clone a single repo. Returns (url, success, message)."""
    dir_name = url_to_dir_name(project_url)
    dest = os.path.join(output_dir, dir_name)

    # skip if already cloned and valid
    if os.path.isdir(dest):
        if subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=dest, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0:
            return (project_url, True, "skipped (exists)")
        # broken repo from interrupted run — remove and re-clone
        shutil.rmtree(dest)

    clone_url = project_url
    if not clone_url.endswith(".git"):
        clone_url += ".git"

    cmd = ["git", "clone", "--quiet", clone_url, dest]
    if not full_clone:
        cmd.insert(2, "--filter=blob:none")

    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=3600, env=GIT_ENV,
        )
        mode = "full" if full_clone else "blobless"
        return (project_url, True, f"cloned ({mode})")
    except subprocess.CalledProcessError as e:
        return (project_url, False, e.stderr.strip())
    except subprocess.TimeoutExpired:
        return (project_url, False, "timed out after 3600s")


def run_clone_phase(repo_commits, output_dir, workers):
    """Phase 1: clone all repos in parallel."""
    urls = sorted(repo_commits.keys())
    total = len(urls)
    n_full = sum(1 for u in urls if len(repo_commits[u]) >= FULL_CLONE_THRESHOLD)
    print(f"Phase 1: Cloning {total} repos ({n_full} full, {total - n_full} blobless) "
          f"with {workers} workers\n", flush=True)

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
                print(f"  [{i}/{total}] {status}: {url_to_dir_name(url)} — {msg}", flush=True)
            if not ok:
                failed_urls.append(url)
            else:
                success += 1

    print(f"\nPhase 1 done. Cloned: {success}, Failed: {len(failed_urls)}", flush=True)
    return failed_urls


# ---------------------------------------------------------------------------
# Phase 2: pre-fetch blobs for dataset commits
# ---------------------------------------------------------------------------

def _fetch_oids(repo_path, oids, deadline=None):
    """Fetch blob OIDs, recursively splitting on E2BIG."""
    if not oids:
        return
    if deadline is None:
        deadline = time.monotonic() + 1800
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    try:
        subprocess.run(
            ["git", "fetch", "origin"] + oids,
            cwd=repo_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=GIT_ENV, timeout=remaining,
        )
    except subprocess.TimeoutExpired:
        repo = os.path.basename(repo_path)
        print(f"  {repo}: fetch timed out ({len(oids)} oids), skipping", flush=True)
    except OSError as e:
        if e.errno == errno.E2BIG and len(oids) > 1:
            mid = len(oids) // 2
            _fetch_oids(repo_path, oids[:mid], deadline)
            _fetch_oids(repo_path, oids[mid:], deadline)
        else:
            raise


def _prefetch_one(project_url, commit_hashes, output_dir, timeout=3600):
    """Pre-fetch blobs for a single repo. Returns (url, num_blobs, message)."""
    dir_name = url_to_dir_name(project_url)
    repo_path = os.path.join(output_dir, dir_name)

    if not os.path.exists(repo_path):
        return (project_url, 0, "repo missing, skipped")

    deadline = time.monotonic() + timeout

    # skip repos that were fully cloned (all blobs already local)
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.promisor"],
        cwd=repo_path, capture_output=True, text=True,
    )
    if result.stdout.strip() != "true":
        return (project_url, 0, "full clone, skipped")

    hashes = list(commit_hashes)
    if not hashes:
        return (project_url, 0, "no commits")

    # Stream diff-tree output to collect blob OIDs without buffering
    proc = None
    try:
        proc = subprocess.Popen(
            ["git", "diff-tree", "-r", "-m", "--stdin", "--no-commit-id"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=repo_path, text=True,
        )

        def _write_stdin():
            try:
                proc.stdin.write("\n".join(hashes))
            except OSError:
                pass
            finally:
                proc.stdin.close()

        writer = threading.Thread(target=_write_stdin, daemon=True)
        writer.start()

        oids = set()
        for line in proc.stdout:
            if time.monotonic() > deadline:
                break
            if not line.startswith(":"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                for oid in (parts[2], parts[3]):
                    if oid != NULL_OID:
                        oids.add(oid)

        if not oids:
            return (project_url, 0, "no blobs needed")

        _fetch_oids(repo_path, list(oids), deadline)
        return (project_url, len(oids), f"fetched {len(oids)} blobs")

    except (subprocess.TimeoutExpired, TimeoutError):
        return (project_url, 0, "timed out")
    finally:
        if proc is not None:
            proc.kill()
            proc.wait()


def run_prefetch_phase(repo_commits, output_dir, workers):
    """Phase 2: pre-fetch blobs for dataset commits in parallel."""
    repos = sorted(repo_commits.keys())
    total = len(repos)
    print(f"\nPhase 2: Pre-fetching blobs for {total} repos with {workers} workers\n", flush=True)

    total_blobs = 0
    failed = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_prefetch_one, url, repo_commits[url], output_dir): url
            for url in repos
        }
        for i, future in enumerate(as_completed(futures), 1):
            url, num_blobs, msg = future.result()
            total_blobs += num_blobs
            with print_lock:
                print(f"  [{i}/{total}] {url_to_dir_name(url)} — {msg}", flush=True)
            if num_blobs == 0 and "skipped" not in msg and "no " not in msg:
                failed.append(url)

    print(f"\nPhase 2 done. Total blobs fetched: {total_blobs}", flush=True)
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

    # Phase 1: clone
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
