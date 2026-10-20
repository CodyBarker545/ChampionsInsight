param(
    [int]$BatchSize = 6,
    [int]$TargetPerVariant = 125,
    [int]$PauseSeconds = 5
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$generator = Join-Path $PSScriptRoot "synthetic_data\generate_synthetic_pokemon_cards.py"
$inputDir = Join-Path $repoRoot "backend\data\pokemon\champions_sprites"
$outputDir = Join-Path $repoRoot "backend\data\training_dataset\slot_pokemon_synthetic"
$logDir = Join-Path $repoRoot "backend\test-output"
$logPath = Join-Path $logDir "synthetic_pokemon_batches.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Discover the full current roster, including both archive pages and shiny variants.
$totalSprites = @(Get-ChildItem -LiteralPath $inputDir -Recurse -Filter "*.png" |
    Where-Object { $_.Name -notmatch 'Mega' }).Count

for ($start = 0; $start -lt $totalSprites; $start += $BatchSize) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] Starting batch at sprite index $start"

    & $python $generator `
        --input $inputDir `
        --output $outputDir `
        --target-per-variant $TargetPerVariant `
        --seed 10 `
        --start-index $start `
        --max-sprites $BatchSize 2>&1 |
        Add-Content -Path $logPath

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] Finished batch at sprite index $start"

    Start-Sleep -Seconds $PauseSeconds
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] All batches complete."
