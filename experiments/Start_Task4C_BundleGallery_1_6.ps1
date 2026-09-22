$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonPath = 'C:/Users/xingdi/sources/FLowlineClusteringVisualAnalysis/.venv/Scripts/python.exe'
$outputRoot = Join-Path $projectRoot 'outputs/Other_Task4C_BundleGallery_1.6'
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
try { $current = Invoke-RestMethod 'http://127.0.0.1:8768/api/status' -TimeoutSec 2 } catch {
    if ($null -ne $_.Exception.Response) { throw 'Port 8768 serves another application; inspect it before replacement.' }
    $current = $null
}
if ($current.version -eq 'Other_Task4C_BundleGallery_1.6') { Write-Output 'Dataset gallery already running.'; exit 0 }
if ($null -ne $current) { throw ('Inspect and stop the prior gallery service before starting 1.6: ' + $current.version) }
$server = Start-Process -FilePath $pythonPath -WorkingDirectory $projectRoot -ArgumentList @('-u','-m','experiments.Serve_Task4C_BundleGallery_1_6') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $outputRoot 'server.stdout.log') -RedirectStandardError (Join-Path $outputRoot 'server.stderr.log') -PassThru
$server.Id | Set-Content -LiteralPath (Join-Path $outputRoot 'server.pid')
Write-Output ('Dataset gallery on 8768, PID ' + $server.Id)
