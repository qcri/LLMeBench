import logging

from pathlib import Path

from git import Repo

DEFAULT_REMOTE_NAME = "origin"
DEFAULT_REMOTE_REPO = "https://github.com/qcri/LLMeBench.git"


def download_all(work_dir):
    if isinstance(work_dir, str):
        work_dir = Path(work_dir)

    repo = Repo.init(work_dir)

    # Create a new remote if there isn't one already created
    if len(repo.remotes) == 0:
        logging.info("Setting up git repo to track assets")
        origin = repo.create_remote(DEFAULT_REMOTE_NAME, DEFAULT_REMOTE_REPO)
        with open(work_dir / ".git/info/sparse-checkout", "w") as fp:
            fp.write("assets\n")
    else:
        # Already in a git repository, checking if this is created by the auto-downloader
        if not (
            repo.remotes[0].name == DEFAULT_REMOTE_NAME
            and repo.remotes[0].url == DEFAULT_REMOTE_REPO
        ):
            logging.error(
                "assets must be downloaded in a clean work_dir that is not already a git repo"
            )
            return False
        else:
            sparse_checkout_file = work_dir / ".git/info/sparse-checkout"

            if not sparse_checkout_file.exists():
                logging.error(
                    "This seems to be a git repository cloned from LLMeBench directly. Please use `git pull` to update your assets."
                )
                return False
            else:
                with open(sparse_checkout_file) as fp:
                    lines = fp.readlines()

                    if len(lines) != 1 or lines[0] != "assets\n":
                        logging.error(
                            "This seems to be a git repository cloned from LLMeBench directly. Please use `git pull` to update your assets."
                        )
                        return False

    logging.info("Downloading/updating assets")
    git = repo.git()
    git.config("core.sparseCheckout", True)
    git.pull("origin", "main")

    return True
