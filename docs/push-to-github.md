# Pushing this project to GitHub

The repository in this folder is **already prepared for you**: it has a `main` branch (empty base)
and a `feat/sms-spam-detector` branch that contains the complete project as one commit. All that is
left is to add your remote and push.

> Why you have to do this last step: the environment where the project was built has **no GitHub
> connection** (no connected GitHub app, no access token, no SSH key), so it cannot authenticate
> against your account. Pushing requires *your* credentials.

---

## Option A — push the prepared repository (recommended)

Unzip the archive, open a terminal in the project folder and run:

```bash
# 0. one-time: put YOUR identity on the commit that is already prepared
git config user.name  "Your Name"
git config user.email "you@example.com"
git commit --amend --reset-author --no-edit        # keeps the message, fixes the author

# 1. connect your repository (create an empty repo on GitHub first - no README, no .gitignore)
git remote add origin https://github.com/<your-username>/<your-repo>.git

# 2. push both branches
git push -u origin main
git push -u origin feat/sms-spam-detector
```

Then open the Pull Request:

* GitHub prints a direct link after the push, **or**
* open `https://github.com/<your-username>/<your-repo>/compare/main...feat/sms-spam-detector`
* set **base: main** ← **compare: feat/sms-spam-detector**, title e.g.
  *"SMS Spam Detector: Django + scikit-learn SMS spam classification"*, and create the PR.

### With the GitHub CLI (fastest)

```bash
gh auth login
gh repo create sms-spam-detector --private --source=. --remote=origin   # or use an existing repo
git push -u origin main
git push -u origin feat/sms-spam-detector
gh pr create --base main --head feat/sms-spam-detector \
  --title "SMS Spam Detector: Django + scikit-learn SMS spam classification" \
  --body "Complete university ML project: TF-IDF pipelines (Naive Bayes, Logistic Regression, Linear SVM), Django web app, Docker setup, 72 tests."
```

### If your repository already has a `main` branch

Skip `git push -u origin main` (you would overwrite it) and push only the feature branch:

```bash
git push -u origin feat/sms-spam-detector
# then open the compare link shown by GitHub
```

---

## Option B — upload the sources without git

1. Create an empty repository on GitHub.
2. Unzip the archive on your machine.
3. On the repository page choose **Add file → Upload files**, drag the *contents* of the
   `sms-spam-detector` folder (not the folder itself, and not the `.git` directory), and commit.
4. GitHub creates the commit directly on the default branch. If you want a PR flow instead, first
   create a branch (**Branch: main → New branch**, e.g. `feat/sms-spam-detector`) and upload there,
   then use **Compare & pull request**.

The web uploader ignores the `.gitignore` file, so remove `data/db.sqlite3` from the list if you
prefer not to publish a local database.

---

## Option C — start a brand-new history

If you would rather not keep the prepared branches:

```bash
cd sms-spam-detector
rm -rf .git
git init -b main
git add .
git commit -m "SMS Spam Detector: Django + scikit-learn SMS spam classification"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

For a PR afterwards: `git checkout -b feat/initial-project`, `git push -u origin feat/initial-project`,
then open `https://github.com/<user>/<repo>/compare/main...feat/initial-project`.

---

## Authentication notes

| Method | How |
| --- | --- |
| HTTPS + Personal Access Token | GitHub → *Settings → Developer settings → Personal access tokens*, scope `repo`. Use the token as the password when git asks (or `git remote set-url origin https://<token>@github.com/<user>/<repo>.git`) |
| SSH | `ssh-keygen -t ed25519 -C "you@example.com"`, add the public key under *Settings → SSH keys*, then `git remote set-url origin git@github.com:<user>/<repo>.git` |
| GitHub CLI / Desktop | `gh auth login` or the GitHub Desktop client handles credentials for you |

Never commit a token into the repository (`.env` and `*.pem` are already ignored by `.gitignore`).

---

## What is inside the prepared commit

* `feat/sms-spam-detector` — 74 files: Django project + app, ML layer, management commands,
  templates, compiled Tailwind CSS, vendored Chart.js, 72 tests, Docker files, documentation, and
  the trained artefacts (`artifacts/`, ≈ 570 kB) so the branch runs right after cloning.
* `main` — a one-file placeholder so the Pull Request has a clean base.

## Good to know

* The dataset (`data/raw/SMSSpamCollection.txt`) is committed because it is the corpus provided with
  the assignment; it is ~460 kB and contains only the SMS messages plus their labels.
* `data/db.sqlite3` is ignored by `.gitignore` and is **not** part of the commit.
* To verify what the branch contains before pushing: `git log --stat feat/sms-spam-detector -1`
  or `git diff --stat main..feat/sms-spam-detector`.
