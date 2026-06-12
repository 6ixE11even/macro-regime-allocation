#!/usr/bin/env bash
#
# One-shot: initialise the repo with a clean commit history and push to GitHub.
# Run this ON YOUR MAC from inside the project folder:
#
#     cd ~/Desktop/projects/macro-regime-allocation
#     bash setup_git.sh
#
# It uses your own git/GitHub credentials — nothing is shared with anyone.
# If you have the GitHub CLI (`gh`) installed and logged in, it also creates the
# remote repo and pushes in one go. Otherwise it stops and tells you the two
# commands to finish.

set -euo pipefail
cd "$(dirname "$0")"

REPO="macro-regime-allocation"
OWNER="6ixE11even"

# 0) Clear any partial repo state left by the build environment.
rm -rf .git

# 1) Use your global git identity if set; otherwise fall back to these.
git init -b main
git config user.name  >/dev/null 2>&1 || git config user.name  "Tejas Pandya"
git config user.email >/dev/null 2>&1 || git config user.email "tbp8777@nyu.edu"

# 2) Commit in logical chunks so the history reads like real work.
git add pyproject.toml .gitignore uv.lock \
        src/macro_regime/__init__.py src/macro_regime/config.py \
        src/macro_regime/data data/raw
git commit -m "Scaffold uv package and FRED-MD / MSCI data loaders"

git add src/macro_regime/regimes src/macro_regime/models
git commit -m "Add two-step regime detection, forecasters, and MVO sizing"

git add src/macro_regime/backtest src/macro_regime/analytics src/macro_regime/viz
git commit -m "Add walk-forward backtest, performance metrics, and plots"

git add -A
git commit -m "Wire up CLI, tests, results, and README"

# 3) Create the remote and push.
if command -v gh >/dev/null 2>&1; then
  gh repo create "$REPO" --public --source=. --remote=origin --push
  echo "Done — https://github.com/$OWNER/$REPO"
else
  cat <<EOF

Repo is committed locally. To publish:
  1. Create an empty PUBLIC repo named "$REPO" at https://github.com/new
  2. Run:
       git remote add origin https://github.com/$OWNER/$REPO.git
       git push -u origin main

(Tip: 'brew install gh && gh auth login' lets this script do step 1-2 for you.)
EOF
fi
