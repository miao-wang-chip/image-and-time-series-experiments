param([ValidateSet('Prepare','Train')][string]$Mode='Prepare', [string]$Python='D:\python\pythonsetup\python.exe')
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$run=Join-Path $project "results/runs/$stamp"
New-Item -ItemType Directory -Path $run -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $project 'src') -Destination (Join-Path $run 'src') -Recurse
Copy-Item -LiteralPath (Join-Path $project 'configs/config.json') -Destination (Join-Path $run 'config.json')
Copy-Item -LiteralPath (Join-Path $project 'data') -Destination (Join-Path $run 'data') -Recurse
New-Item -ItemType Directory -Path (Join-Path $run 'results') -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $project 'results') -Force | Where-Object { $_.Name -ne 'runs' } | Copy-Item -Destination (Join-Path $run 'results') -Recurse
New-Item -ItemType Directory -Path (Join-Path $run 'docs') -Force | Out-Null
Write-Host "Independent run workspace: $run"
if ($Mode -eq 'Prepare') { return }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python interpreter not found; pass -Python.' }
Push-Location $run
try {
    & $Python -u src/run.py
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $LASTEXITCODE" }
} finally { Pop-Location }
