param([int]$Port = 8767)

$analysisOutput = Join-Path $PSScriptRoot 'outputs'
$analysisPage = Join-Path $analysisOutput 'Other_FMT_AnalysisWorkbench_1.1/index.html'
if (-not (Test-Path -LiteralPath $analysisPage)) {
    throw 'Build the viewer first: python -m experiments.Build_FMT_FeatureContrast_1_1'
}
$analysisUrl = "http://127.0.0.1:$Port/Other_FMT_AnalysisWorkbench_1.1/index.html"
Write-Host "FMT analysis: $analysisUrl"
Write-Host 'This local server stays in this terminal. Press Ctrl+C to stop.'
python -m http.server $Port --bind 127.0.0.1 --directory $analysisOutput
