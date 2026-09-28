# Flattens each split's loose *.jpg / *.txt into images\ and labels\ subfolders.
# Paths resolve from the repo root, so it works from any working directory.

$repoRoot = Split-Path -Parent $PSScriptRoot
$datasetRoot = Join-Path $repoRoot "data\license_plate_dataset"

foreach ($split in @("train","val","test")) {
    $splitPath = Join-Path $datasetRoot $split
    $imgDir = Join-Path $splitPath "images"
    $lblDir = Join-Path $splitPath "labels"
    New-Item -ItemType Directory -Force -Path $imgDir | Out-Null
    New-Item -ItemType Directory -Force -Path $lblDir | Out-Null
    Get-ChildItem -Path $splitPath -Filter "*.jpg" -File | Move-Item -Destination $imgDir -Force
    Get-ChildItem -Path $splitPath -Filter "*.txt" -File | Move-Item -Destination $lblDir -Force
    Write-Host "[$split] Moved $((Get-ChildItem $imgDir).Count) images and $((Get-ChildItem $lblDir).Count) labels."
}
