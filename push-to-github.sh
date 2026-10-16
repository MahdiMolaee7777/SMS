#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command push of the prepared SMS Spam Detector repository to a GitHub
# repository you own, followed by the Pull Request link.
#
#   ./push-to-github.sh <repo-url> [author-name] [author-email]
#
# Examples
#   ./push-to-github.sh https://github.com/ali/sms-spam-detector.git
#   ./push-to-github.sh git@github.com:ali/sms-spam-detector.git "Ali" "ali@example.com"
#
# What it does
#   1. checks that you are inside the prepared repository,
#   2. (optionally) puts your name on the prepared commit,
#   3. points `origin` at your repository,
#   4. pushes `main` and `feat/sms-spam-detector`,
#   5. prints the compare/PR link.
#
# Authentication is yours: use a Personal Access Token when git asks for a
# password over HTTPS, or an SSH key with a git@github.com URL. The script never
# stores or prints credentials.
# ---------------------------------------------------------------------------
set -euo pipefail

FEATURE_BRANCH="feat/sms-spam-detector"
BASE_BRANCH="main"

die() { printf '\033[31m✖ %s\033[0m\n' "$1" >&2; exit 1; }
info() { printf '\033[36m▸ %s\033[0m\n' "$1"; }
ok() { printf '\033[32m✔ %s\033[0m\n' "$1"; }

REPO_URL="${1:-}"
AUTHOR_NAME="${2:-}"
AUTHOR_EMAIL="${3:-}"

[ -n "$REPO_URL" ] || die "Usage: ./push-to-github.sh <repo-url> [author-name] [author-email]"
command -v git >/dev/null || die "git is not installed."
git rev-parse --git-dir >/dev/null 2>&1 || die "Run this script inside the sms-spam-detector folder (no git repository found)."
git rev-parse --verify "$FEATURE_BRANCH" >/dev/null 2>&1 || die "Branch '$FEATURE_BRANCH' not found - is this the prepared repository?"

# --- 1. author identity ----------------------------------------------------
if [ -n "$AUTHOR_NAME" ] && [ -n "$AUTHOR_EMAIL" ]; then
  info "Setting the commit author to $AUTHOR_NAME <$AUTHOR_EMAIL>"
  git config user.name "$AUTHOR_NAME"
  git config user.email "$AUTHOR_EMAIL"
  git checkout -q "$FEATURE_BRANCH"
  git commit -q --amend --reset-author --no-edit
  git checkout -q "$FEATURE_BRANCH"
  ok "Prepared commit now carries your identity."
elif [ -z "$(git config user.email || true)" ]; then
  info "No git identity configured. Re-run with:  ./push-to-github.sh <repo-url> \"Your Name\" \"you@example.com\""
fi

# --- 2. remote -------------------------------------------------------------
if git remote get-url origin >/dev/null 2>&1; then
  info "Replacing existing remote 'origin'"
  git remote remove origin
fi
git remote add origin "$REPO_URL"
ok "origin -> $REPO_URL"

# --- 3. push ---------------------------------------------------------------
info "Pushing '$BASE_BRANCH' ..."
if git push -u origin "$BASE_BRANCH"; then
  ok "'$BASE_BRANCH' pushed."
else
  printf '\033[33m! Could not push %s (it may already contain commits on the remote).\033[0m\n' "$BASE_BRANCH"
  printf '\033[33m  Continuing with the feature branch so nothing is overwritten.\033[0m\n'
fi

info "Pushing '$FEATURE_BRANCH' (the complete project) ..."
git push -u origin "$FEATURE_BRANCH"
ok "'$FEATURE_BRANCH' pushed."

# --- 4. pull request link --------------------------------------------------
WEB_URL="$(printf '%s' "$REPO_URL" \
  | sed -e 's#^git@github\.com:#https://github.com/#' -e 's#\.git$##')"
BASE_COMPARE="$BASE_BRANCH"
git rev-parse --verify "origin/$BASE_BRANCH" >/dev/null 2>&1 || BASE_COMPARE=""

printf '\n'
ok "Done. Everything is on GitHub."
printf '\nCreate the Pull Request here:\n'
if [ -n "$BASE_COMPARE" ]; then
  printf '  \033[1m%s/compare/%s...%s?expand=1\033[0m\n' "$WEB_URL" "$BASE_COMPARE" "$FEATURE_BRANCH"
else
  printf '  \033[1m%s/pulls\033[0m  (GitHub proposes the branch automatically)\n' "$WEB_URL"
fi
printf '\nVerification:\n'
printf '  %s/branches           -> %s and %s\n' "$WEB_URL" "$BASE_BRANCH" "$FEATURE_BRANCH"
printf '  %s/tree/%s            -> all project files\n' "$WEB_URL" "$FEATURE_BRANCH"
