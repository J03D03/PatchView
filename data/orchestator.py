import csv
import json
import os
import random
import subprocess
import git
import traceback
from urllib.parse import urlparse
from data.misc import tqdm
import pathlib
from pydriller import Repository

SEED = 0x1337

repo_commits_location = r"cache_data\repo_commits.json"
MAXIMAL_FILE_SIZE = 100000


TRAIN_RATE = 1
VALIDATION_RATE = 0
TEST_RATE = 0


def get_benign_commits(cur_repo_path,repo, security_commits):
    cur_repo = cur_repo_path + "/" + repo.replace("/", "_")
    try:
        mall = subprocess.run(
            ["git", "rev-list", "--all", "--since=2015"],
            stdout=subprocess.PIPE,
            cwd=cur_repo,
            timeout=300,
            check=True,
        )
        mall = mall.stdout.decode("utf-8").split("\n")
    except subprocess.TimeoutExpired:
        print(f"Timed out running git rev-list for {cur_repo}")
        return
    except subprocess.CalledProcessError as e:
        print(f"Failed to run git rev-list for {cur_repo}: {e}")
        return
    random.shuffle(mall)
    for commit in mall:
        if commit in security_commits:
            continue

        # pydriller_commit = get_commit_from_repo(cur_repo, commit)
        # if len(pydriller_commit.modified_files) == 0:
        #     continue

        yield commit
    return


def add_code_data_to_dict(file):
    cur_dict = {}
    before = ""
    after = ""
    if file.content_before is not None:
        try:
            before = file.content_before.decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        before = ""

    if file.content is not None:
        try:
            after = file.content.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if "." not in file.filename:
        return None
    if len(after) > MAXIMAL_FILE_SIZE or len(before) > MAXIMAL_FILE_SIZE:
        return None

    filetype = file.filename.split(".")[-1].lower()
    cur_dict["filetype"] = filetype
    cur_dict["filename"] = file.filename
    cur_dict["content"] = after
    cur_dict["before_content"] = before
    cur_dict["added"] = file.diff_parsed["added"]
    cur_dict["deleted"] = file.diff_parsed["deleted"]
    return cur_dict


def get_commit_from_repo(cur_repo, hash):
    from pydriller import Repository

    res = next(Repository(cur_repo, single=hash).traverse_commits())
    return res


def prepare_dict(cur_repo_path,repo, commit, label):
    try:
        commit = get_commit_from_repo(
            os.path.join(cur_repo_path, repo.replace("/", "_")), commit
        )
        final_dict = {}
        final_dict["name"] = commit.project_name
        final_dict["hash"] = commit.hash
        final_dict["files"] = []
        final_dict["source"] = []
        final_dict["label"] = label
        final_dict["repo"] = repo
        final_dict["message"] = commit.msg
        if len(commit.modified_files) == 0:
            raise CommitNotFound
        for file in commit.modified_files:
            cur_dict = add_code_data_to_dict(file)
            if cur_dict is not None:
                final_dict["files"].append(cur_dict)
    except Exception:
        raise CommitNotFound

    return final_dict


def get_orchestrator_from_details(train_details, val_details, test_details, cache_path=None, cur_repo_path="/storage/nitzan/dataset/commits"):
    random.seed(SEED)

    mall = {}
    for row in train_details:
        mall[row[3]] = {"label": row[2], "repo": row[0]}
    for row in val_details:
        mall[row[3]] = {"label": row[2], "repo": row[0]}
    for row in test_details:
        mall[row[3]] = {"label": row[2], "repo": row[0]}

    new_mall = {}
    repo_set = dict()
    counter = 0
    for a, b in tqdm(mall.items()):
        counter += 1
        new_mall[a] = b
        if b["repo"] not in repo_set:
            commit = get_benign_commits(cur_repo_path, b["repo"], mall.keys())
            repo_set[b["repo"]] = commit
        for _ in range(2):
            try:
                commit_hash = next(repo_set[b["repo"]])
                if commit_hash in new_mall:
                    print(f"Already found {commit_hash}-{b['repo']}-{0}")
                    continue
                new_mall[commit_hash] = {"label": 0, "repo": b["repo"]}
            except StopIteration:
                continue

    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump(new_mall, f, indent=4)

    return new_mall


class CommitNotFound(Exception):
    """Raised when the commit is not found"""

    pass


def split_randomly(cur_repo_path, data):
    commit_dict = {}
    data_keys = list(data)[:]
    err_list = []

    all_commits = []
    for repo in (pbar:=tqdm(data_keys)):
        pbar.set_description(f"Processing {repo}")

        positive_repo_counter = 0
        negative_repo_counter = 0
        for commit in data[repo]:
            if commit == "":
                continue
            try:
                commit_dict[commit] = prepare_dict(cur_repo_path, repo, commit, 1)
                positive_repo_counter += 1
            except CommitNotFound:
                err_list.append((repo, commit, "CommitNotFound"))
                continue

        commit = get_benign_commits(cur_repo_path, repo, data[repo])
        while negative_repo_counter < positive_repo_counter:
            try:
                current_commit = next(commit)
            except StopIteration:
                break

            commit_hash = current_commit
            try:
                commit_dict[commit_hash] = prepare_dict(cur_repo_path, repo, commit_hash, 0)
                negative_repo_counter += 1
            except CommitNotFound:
                continue

    random.shuffle(all_commits)
    print(err_list)

    return commit_dict


def split_by_repos(cur_repo_path, data):
    data_keys = list(data)[:]
    # random.shuffle(data_keys)

    num_of_repos = len(data_keys)
    num_of_repos_training = int(num_of_repos * TRAIN_RATE)
    num_of_repos_validation = int(num_of_repos * VALIDATION_RATE)
    num_of_repos_testing = int(num_of_repos * TEST_RATE)

    training_keys = data_keys[:num_of_repos_training]
    validation_keys = data_keys[
        num_of_repos_training : num_of_repos_training + num_of_repos_validation
    ]
    testing_keys = data_keys[num_of_repos_training + num_of_repos_validation :]

    print(f"Training size: {num_of_repos_training}")
    print(f"Validation size: {num_of_repos_validation}")
    print(f"Testing size: {num_of_repos_testing}")

    training_dict = {}
    validation_dict = {}
    testing_dict = {}

    for dict, keys in zip(
        [training_dict, validation_dict, testing_dict],
        [training_keys, validation_keys, testing_keys],
    ):
        for repo in tqdm(keys, desc="Splitting by repos"):
            try:
                dict[repo] = []
                for commit in data[repo]:
                    if commit == "":
                        continue
                    dict[repo].append((commit, 1))

                for commit in get_benign_commits(cur_repo_path, repo, data[repo]):
                    dict[repo].append((commit, 0))
            except git.exc.NoSuchPathError as e:
                print(e)
            except ValueError as e:
                print(e)
    return training_dict, validation_dict, testing_dict



def get_orchestrator(commits_path, commits_db_path, cache_path=None,should_split_by_repos=False):
    random.seed(SEED)

    with open(commits_db_path, "r") as fin:
        repo_commits = json.load(fin)


    if should_split_by_repos:
        return split_by_repos(commits_path, repo_commits)
    else:
        return split_randomly(commits_path, repo_commits)



def url_to_repo_name(project_url):
    """Convert a project URL to a repo name (path portion after the domain).

    Examples:
        https://github.com/01org/opa-ff -> 01org/opa-ff
        https://android.googlesource.com/platform/dalvik -> platform/dalvik
    """
    parsed = urlparse(project_url)
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path


def _load_csv_split(csv_path, mall, hash_list):
    """Read a single CSV split file and populate mall and hash_list."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            commit_id = row["commit_id"]
            label = 1 if row["is_vfc"] == "True" else 0
            repo = url_to_repo_name(row["project_url"])
            mall[commit_id] = {"label": label, "repo": repo}
            hash_list.append(commit_id)


def get_orchestrator_from_csv(csv_train_path, csv_val_path, csv_test_path):
    """Load pre-defined train/val/test splits from CSV files.

    CSV format: project_url,commit_id,is_vfc,commit_timestamp_utc

    Returns:
        mall: dict mapping commit_hash -> {"label": int, "repo": str}
        train_hashes: list of training commit hashes
        val_hashes: list of validation commit hashes
        test_hashes: list of test commit hashes
    """
    mall = {}
    train_hashes = []
    val_hashes = []
    test_hashes = []

    _load_csv_split(csv_train_path, mall, train_hashes)
    _load_csv_split(csv_val_path, mall, val_hashes)
    _load_csv_split(csv_test_path, mall, test_hashes)

    print(f"Loaded CSV splits — train: {len(train_hashes)}, "
          f"val: {len(val_hashes)}, test: {len(test_hashes)}")

    return mall, train_hashes, val_hashes, test_hashes


if __name__ == "__main__":
    with open(r"C:\Users\nitzan\local\analyzeCVE\last_train.json", "r") as mfile:
        train_details = json.load(mfile)
    with open(r"C:\Users\nitzan\local\analyzeCVE\last_val.json", "r") as mfile:
        val_details = json.load(mfile)
    with open(r"C:\Users\nitzan\local\analyzeCVE\last_test.json", "r") as mfile:
        test_details = json.load(mfile)
    get_orchestrator_from_details(train_details, val_details, test_details, cache_path=os.path.join("cache_data", "orc", "orchestrator.json"))
