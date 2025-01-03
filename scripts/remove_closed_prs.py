#!/usr/bin/env python3

from glob import glob
import os.path as op
import shutil
import subprocess
import argparse
from create_collection_md import check_single_source

API_CALL = "curl 'https://api.github.com/repos/{repo}/pulls/{num}' | jq '.closed_at'"

def main(git_filter_repo=False, git_filter_repo_command=None):
    """Remove trees corresponding to closed PRs.

    Behavior depends on git_filter_repo:

    - if False, then we call shutil.rmtree to just delete the subtree corresponding to
      the pull request

    - if True, we use git filter-repo to remove the subtree from the git history

    """
    output_str = ""
    if git_filter_repo:
        output_str = " from git history"
    if git_filter_repo_command is None:
        git_filter_repo_command = "git filter-repo"
    if check_single_source():
        sites = ['workshops/']
    else:
        sites = glob('workshops/*/')
    # throughout this, we do something like `op.split(variable[:-1])[-1]` to grab the folder
    # name. that's because we glob strings that end in /, so we need to drop that before
    # using op.split
    for site in sites:
        with open(op.join(site, '.gh_path')) as f:
            repo = f.read().strip()
        for pull in glob(op.join(site, 'pulls/*/')):
            num = op.split(pull[:-1])[-1]
            response = subprocess.run(API_CALL.format(repo=repo, num=num), shell=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            response = response.stdout.decode('utf-8').strip()
            if response != 'null':
                print(f"{repo} PR {num} is closed, removing its docs{output_str} at {pull}!")
                if git_filter_repo:
                    subprocess.run(f"{git_filter_repo_command} --invert-paths --path {pull}",
                                   shell=True)
                else:
                    shutil.rmtree(pull)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Delete subtrees corresponding to closed PRs."
    )
    parser.add_argument("--git_filter_repo", "-g", action="store_true",
                        help=("Remove subtree from git history (using git-filter-repo), "
                              "instead of only deleting from disk."))
    parser.add_argument("--git_filter_repo_command", "--cmd", default=None,
                        help=("Command for git filter repo. If unset, will use `git filter-repo`,"
                              " which assumes you have it on your path"))
    args = vars(parser.parse_args())
    main(**args)
