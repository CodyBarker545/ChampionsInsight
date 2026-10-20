param(
    [string]$ReleaseName = "ChampionsInsight-test-machine"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$DistRoot = Join-Path $ProjectRoot "dist"
$StageRoot = Join-Path $DistRoot ("release-stage-" + [System.Guid]::NewGuid().ToString("N"))
$ReleaseRoot = Join-Path $StageRoot $ReleaseName
$ZipPath = Join-Path $DistRoot "$ReleaseName.zip"

function Assert-PathInsideProject {
    param([Parameter(Mandatory = $true)][string]$PathToCheck)

    $projectFull = [System.IO.Path]::GetFullPath($ProjectRoot.Path)
    $targetFull = [System.IO.Path]::GetFullPath($PathToCheck)
    if (-not $targetFull.StartsWith($projectFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside the project root: $targetFull"
    }
}

function Copy-ReleasePath {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [string[]]$Exclude = @()
    )

    $source = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source)) {
        Write-Host "Skipping missing path: $RelativePath"
        return
    }

    $destination = Join-Path $ReleaseRoot $RelativePath
    $destinationParent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null

    if (Test-Path -LiteralPath $source -PathType Container) {
        New-Item -ItemType Directory -Force -Path $destination | Out-Null
        $robocopyArgs = @(
            $source,
            $destination,
            "/E",
            "/NFL",
            "/NDL",
            "/NJH",
            "/NJS",
            "/NC",
            "/NS",
            "/XD",
            "__pycache__",
            "node_modules",
            "dist",
            ".pytest_cache",
            "/XF",
            "*.pyc",
            "*.pyo"
        )
        & robocopy @robocopyArgs | Out-Null
        if ($LASTEXITCODE -gt 7) {
            throw "robocopy failed for ${RelativePath} with exit code $LASTEXITCODE"
        }
        return
    }

    Copy-Item -LiteralPath $source -Destination $destination -Force
}

Assert-PathInsideProject $DistRoot
Assert-PathInsideProject $StageRoot
Assert-PathInsideProject $ReleaseRoot
Assert-PathInsideProject $ZipPath

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
New-Item -ItemType Directory -Force -Path $ReleaseRoot | Out-Null

$pathsToCopy = @(
    ".gitignore",
    "README.md",
    "requirements.txt",
    "setup_test_machine.ps1",
    "start_champions_insight.ps1",
    "backend\README.md",
    "backend\app.py",
    "backend\paths.py",
    "backend\requirements.txt",
    "backend\api",
    "backend\rag",
    "backend\RAGDocs",
    "backend\services",
    "backend\scripts",
    "backend\data\README.md",
    "backend\data\championsinsight_archetype_notes.json",
    "backend\data\pokemon",
    "backend\data\rag",
    "backend\data\competitive\vgc_data",
    "backend\data\competitive\analysis_json\archetype_cluster_summary.json",
    "backend\data\competitive\analysis_json\archetype_rule_summary.json",
    "backend\data\competitive\analysis_json\conditional_set_probability_summary.json",
    "backend\data\competitive\analysis_json\pokedex_meta_intelligence.json",
    "backend\data\competitive\analysis_json\pokedex_usage_summary.json",
    "backend\data\competitive\analysis_json\pokemon_recommended_builds.json",
    "backend\data\competitive\analysis_json\pokemon_top_6_moves.json",
    "backend\data\competitive\analysis_json\pokemon_top_3_items.json",
    "backend\data\competitive\analysis_json\archetype_team_classifications.json",
    "backend\data\competitive\analysis_json\set_probability_summary.json",
    "backend\data\cv\models\pokemon_mobilenetv3\pokemon_mobilenetv3.pt",
    "backend\data\cv\models\pokemon_mobilenetv3\idx_to_class.json",
    "backend\data\cv\models\pokemon_mobilenetv3\class_to_idx.json",
    "backend\data\cv\models\slot_object_detector\yolov8n_slot_objects_padded_clean_types\weights\best.pt",
    "backend\vector_store",
    "backend\pokemon_damage_calculator\__init__.py",
    "backend\pokemon_damage_calculator\old",
    "backend\pokemon_damage_calculator\backend_process.py",
    "backend\pokemon_damage_calculator\damage-calc-master\package.json",
    "backend\pokemon_damage_calculator\damage-calc-master\package-lock.json",
    "backend\pokemon_damage_calculator\damage-calc-master\server.js",
    "backend\pokemon_damage_calculator\damage-calc-master\build",
    "backend\pokemon_damage_calculator\damage-calc-master\bundler.js",
    "backend\pokemon_damage_calculator\damage-calc-master\publish",
    "backend\pokemon_damage_calculator\damage-calc-master\src",
    "backend\pokemon_damage_calculator\damage-calc-master\calc\package.json",
    "backend\pokemon_damage_calculator\damage-calc-master\calc\package-lock.json",
    "backend\pokemon_damage_calculator\damage-calc-master\calc\bundle",
    "backend\pokemon_damage_calculator\damage-calc-master\calc\dist",
    "backend\pokemon_damage_calculator\damage-calc-master\calc\src",
    "frontend\index.html",
    "frontend\package.json",
    "frontend\package-lock.json",
    "frontend\vite.config.js",
    "frontend\src"
)

foreach ($path in $pathsToCopy) {
    Copy-ReleasePath $path -Exclude @("__pycache__", "*.pyc", "node_modules", "dist", ".pytest_cache")
}

$runtimeDirs = @(
    "backend\data\uploads",
    "backend\data\user",
    "backend\data\cv\debug\crops",
    "backend\data\cv\debug\reports"
)

foreach ($relativeDir in $runtimeDirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $ReleaseRoot $relativeDir) | Out-Null
}

$releaseNotes = @"
# Champions Insight Test-Machine Package

Run this from the extracted folder:

```powershell
.\setup_test_machine.ps1
```

This installs Python and Node dependencies into local project folders and creates
a Desktop shortcut named "Champions Insight".

Then start the app with the shortcut, or run:

```powershell
.\start_champions_insight.ps1
```

This package ships the prebuilt YOLO, MobileNetV3, RAG, and current tournament
meta artifacts needed at runtime. Legacy CV indexes and reference-image libraries
are excluded. It also excludes local
uploads, training datasets, debug images, test caches, virtual environments,
node_modules, videos, and generated photo sets. Uploaded team photos and opponent
prediction results are transient runtime state only and are not saved to disk.
"@

Set-Content -LiteralPath (Join-Path $ReleaseRoot "RELEASE_README.md") -Value $releaseNotes -Encoding UTF8

Compress-Archive -LiteralPath $ReleaseRoot -DestinationPath $ZipPath -CompressionLevel Optimal

$zipItem = Get-Item -LiteralPath $ZipPath
Write-Host "Created $($zipItem.FullName)"
Write-Host "Size: $([math]::Round($zipItem.Length / 1MB, 2)) MB"

try {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force
}
catch {
    Write-Warning "Could not remove temporary staging folder: $StageRoot"
}
