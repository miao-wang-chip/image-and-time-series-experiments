param([ValidateSet('Prepare','Train')][string]$Mode='Prepare', [string]$Python='D:\python\pythonsetup\python.exe')
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$run=Join-Path $project "results/runs/$stamp"
New-Item -ItemType Directory -Path $run -Force | Out-Null
Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.py' -File | Copy-Item -Destination $run
Copy-Item -LiteralPath (Join-Path $project 'data') -Destination (Join-Path $run 'data') -Recurse
Copy-Item -LiteralPath (Join-Path $project 'checkpoints') -Destination (Join-Path $run 'weights') -Recurse
New-Item -ItemType Directory -Path (Join-Path $run 'results') -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $project 'results') -Force | Where-Object { $_.Name -ne 'runs' } | Copy-Item -Destination (Join-Path $run 'results') -Recurse
New-Item -ItemType Directory -Path (Join-Path $run 'docs') -Force | Out-Null
Write-Host "Independent run workspace: $run"
if ($Mode -eq 'Prepare') { return }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python interpreter not found; pass -Python.' }
Push-Location $run
try {
    & $Python -u run_task1.py
    if ($LASTEXITCODE -ne 0) { throw "Task 1 failed: $LASTEXITCODE" }
    & $Python -u run_task2.py
    if ($LASTEXITCODE -ne 0) { throw "Task 2 failed: $LASTEXITCODE" }
} finally { Pop-Location }
