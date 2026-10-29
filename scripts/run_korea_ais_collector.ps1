$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$collectorPath = Join-Path $repoRoot "collect_ais_to_csv.py"
$outputPath = Join-Path $repoRoot "data\ais\korea_ais_stream.csv"
$logDirectory = Join-Path $repoRoot "logs"
$logPath = Join-Path $logDirectory "korea_ais_collector.log"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
Set-Location -LiteralPath $repoRoot

& $pythonPath -u $collectorPath `
    --region korea `
    --output $outputPath `
    --max-messages 0 *>> $logPath
