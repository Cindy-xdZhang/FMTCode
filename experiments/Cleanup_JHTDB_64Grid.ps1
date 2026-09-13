$ErrorActionPreference = 'Stop'
$workspacePath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$oldRoot = (Resolve-Path -LiteralPath (Join-Path $workspacePath 'outputs\Verify_JHTDB_VTKDownload_1.1')).Path
$newRoot = (Resolve-Path -LiteralPath (Join-Path $workspacePath 'outputs\Verify_JHTDB_DualFormatDownload_1.2')).Path
if (-not $oldRoot.StartsWith($workspacePath + '\outputs\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Old data must be inside workspace outputs' }
$audit = Get-Content -LiteralPath (Join-Path $newRoot 'audit.json') -Raw | ConvertFrom-Json
if ($audit.status -ne 'pass') { throw 'New audit did not pass' }
$targets = @()
foreach ($name in @('channel','isotropic')) {
    $result = $audit.flows.$name
    if ($result.frames -ne 32 -or $result.netcdf.status -ne 'pass' -or -not $result.netcdf.all_coordinates_times_velocities_equal_vtk) { throw 'Both new formats must pass all 32 frames' }
    $newManifest = Get-Content -LiteralPath (Join-Path $newRoot "$name\manifest.json") -Raw | ConvertFrom-Json
    if ($newManifest.config.x_res -ne 128 -or $newManifest.config.y_res -ne 128 -or $newManifest.config.z_res -ne 128) { throw 'Replacement resolution must be 128 cubed' }
    $oldDirectory = (Resolve-Path -LiteralPath (Join-Path $oldRoot $name)).Path
    $m = Get-Content -LiteralPath (Join-Path $oldDirectory 'manifest.json') -Raw | ConvertFrom-Json
    foreach ($r in $m.frames) {
        if ($r.file -notmatch ('^' + $name + '_\d{3}\.vtk$')) { throw 'Unexpected old frame filename' }
        $targets += [pscustomobject]@{ path=(Join-Path $oldDirectory $r.file); expected_sha256=$r.sha256; parent=$oldDirectory }
    }
    if ($m.combined.file -ne "$name.vtk") { throw 'Unexpected old combined filename' }
    $targets += [pscustomobject]@{ path=(Join-Path $oldDirectory $m.combined.file); expected_sha256=$m.combined.sha256; parent=$oldDirectory }
    foreach ($derived in @("$name.vtk.series",'flow_3d.html')) {
        $targetPath = Join-Path $oldDirectory $derived
        if (Test-Path -LiteralPath $targetPath) {
            $targets += [pscustomobject]@{ path=$targetPath; expected_sha256=(Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant(); parent=$oldDirectory }
        }
    }
}
if (($targets | Where-Object { $_.path.EndsWith('.vtk') }).Count -ne 66) { throw 'Expected 66 old VTK files' }
$firstRoot = (Resolve-Path -LiteralPath (Join-Path $workspacePath 'outputs\Verify_JHTDB_ChannelDownload_1.1')).Path
$firstPreview = Join-Path $firstRoot 'channel_3d.html'
if (Test-Path -LiteralPath $firstPreview) {
    $targets += [pscustomobject]@{ path=$firstPreview; expected_sha256=(Get-FileHash -LiteralPath $firstPreview -Algorithm SHA256).Hash.ToLowerInvariant(); parent=$firstRoot }
}
# Validate all resolved paths and hashes before deleting any exact file. No recursive deletion.
foreach ($entry in $targets) {
    $resolved = (Resolve-Path -LiteralPath $entry.path).Path
    $inOldDataset = $resolved.StartsWith($oldRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or $resolved -eq $firstPreview
    if ([IO.Path]::GetDirectoryName($resolved) -ne $entry.parent -or -not $inOldDataset) { throw 'Deletion target escapes the old dataset directory' }
    if ((Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.expected_sha256) { throw "Old checksum changed: $resolved" }
}
$report = [ordered]@{ status='verified_before_deletion'; reason='User requested new 128 cubed downloads, doubled spans, VTK and complete NetCDF'; files=$targets }
$reportPath = Join-Path $newRoot 'old_grid_cleanup.json'
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding utf8
foreach ($entry in $targets) { Remove-Item -LiteralPath $entry.path -Force }
foreach ($entry in $targets) { if (Test-Path -LiteralPath $entry.path) { throw 'Deletion failed' } }
$report.status = 'complete'
$report.completed_utc = [DateTime]::UtcNow.ToString('o')
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding utf8
Write-Output 'Deleted 66 old VTK files and their series/previews; preserved manifests, audits, configs and source records.'
