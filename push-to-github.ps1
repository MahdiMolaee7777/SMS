# ---------------------------------------------------------------------------
# Windows / PowerShell equivalent of push-to-github.sh
#
#   .\push-to-github.ps1 -RepoUrl https://github.com/you/sms-spam-detector.git
#   .\push-to-github.ps1 -RepoUrl git@github.com:you/sms-spam-detector.git `
#                        -AuthorName "Your Name" -AuthorEmail "you@example.com"
#
# If PowerShell blocks the script, run once:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# ---------------------------------------------------------------------------
param(
    [Parameter(Mandatory = $true)][string]$RepoUrl,
    [string]$AuthorName = "",
    [string]$AuthorEmail = ""
)

$ErrorActionPreference = "Stop"
$FeatureBranch = "feat/sms-spam-detector"
$BaseBranch = "main"

function Info($m) { Write-Host "> $m" -ForegroundColor Cyan }
function Ok($m) { Write-Host "OK  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "! $m" -ForegroundColor Yellow }

if (-not (Test-Path ".git")) {
    throw "Run this script inside the sms-spam-detector folder (no git repository found)."
}
& git rev-parse --verify $FeatureBranch *> $null
if ($LASTEXITCODE -ne 0) { throw "Branch '$FeatureBranch' not found - is this the prepared repository?" }

if ($AuthorName -and $AuthorEmail) {
    Info "Setting the commit author to $AuthorName <$AuthorEmail>"
    & git config user.name $AuthorName
    & git config user.email $AuthorEmail
    & git checkout -q $FeatureBranch
    & git commit -q --amend --reset-author --no-edit
    Ok "Prepared commit now carries your identity."
}

$existing = (& git remote 2>$null)
if ($existing -contains "origin") {
    Info "Replacing existing remote 'origin'"
    & git remote remove origin
}
& git remote add origin $RepoUrl
Ok "origin -> $RepoUrl"

Info "Pushing '$BaseBranch' ..."
& git push -u origin $BaseBranch
if ($LASTEXITCODE -ne 0) {
    Warn "Could not push $BaseBranch (it may already contain commits). Continuing with the feature branch."
} else {
    Ok "$BaseBranch pushed."
}

Info "Pushing '$FeatureBranch' (the complete project) ..."
& git push -u origin $FeatureBranch
if ($LASTEXITCODE -ne 0) { throw "Push of the feature branch failed - check your credentials and repository name." }
Ok "$FeatureBranch pushed."

$WebUrl = $RepoUrl -replace '^git@github\.com:', 'https://github.com/' -replace '\.git$', ''
Write-Host ""
Ok "Done. Everything is on GitHub."
Write-Host ""
Write-Host "Create the Pull Request here:" -ForegroundColor White
Write-Host "  $WebUrl/compare/$BaseBranch...$FeatureBranch?expand=1" -ForegroundColor White
Write-Host ""
Write-Host "Verification:" -ForegroundColor White
Write-Host "  $WebUrl/branches"
Write-Host "  $WebUrl/tree/$FeatureBranch"
