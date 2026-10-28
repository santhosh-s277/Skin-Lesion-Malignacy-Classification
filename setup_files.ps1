# Edit these two names ONLY if your browser saved the files under different names
$tarName = "ISIC2019_valtest_cache256.tar"
$zipName = "splits.zip"

$root     = "C:\SafeDerm"
$dl       = Join-Path $env:USERPROFILE "Downloads"
$tarSrc   = Join-Path $dl $tarName
$zipSrc   = Join-Path $dl $zipName
$splitDir = Join-Path $root "splits"
$tarDst   = Join-Path $root "ISIC2019_valtest_cache256.tar"
$tmpDir   = Join-Path $root "_tmp_splits_unzip"

if (-not (Test-Path $root)) { Write-Host "STOP: $root does not exist."; return }
Write-Host "Downloads folder : $dl"
Write-Host ("tar in Downloads : " + (Test-Path $tarSrc))
Write-Host ("zip in Downloads : " + (Test-Path $zipSrc))

# 1. tar: copied as-is, NOT extracted. Nothing already in C:\SafeDerm is overwritten.
if (Test-Path $tarDst) { Write-Host "tar already in C:\SafeDerm, left untouched." }
elseif (Test-Path $tarSrc) { Copy-Item $tarSrc $tarDst; Write-Host "tar copied." }
else { Write-Host "tar not found in Downloads, skipped." }

# 2. splits: unzip to a temp folder, find the 3 CSVs wherever they are, copy them flat
New-Item -ItemType Directory -Force -Path $splitDir | Out-Null
$need = @("train.csv","val.csv","test.csv") | Where-Object { -not (Test-Path (Join-Path $splitDir $_)) }
$need = @($need)
if ($need.Count -eq 0) { Write-Host "splits already in place, left untouched." }
elseif (Test-Path $zipSrc) {
    if (Test-Path $tmpDir) { Remove-Item $tmpDir -Recurse -Force }
    Expand-Archive -Path $zipSrc -DestinationPath $tmpDir
    foreach ($f in $need) {
        $hit = Get-ChildItem -Path $tmpDir -Recurse -Filter $f -File | Select-Object -First 1
        if ($hit) { Copy-Item $hit.FullName (Join-Path $splitDir $f); Write-Host "copied $f" }
        else { Write-Host "NOT FOUND inside the zip: $f" }
    }
    Remove-Item $tmpDir -Recurse -Force
}
else { Write-Host "splits zip not found in Downloads, skipped." }

# 3. verification
Write-Host "`n===== VERIFICATION (all four must be True) ====="
foreach ($p in @("$splitDir\train.csv","$splitDir\val.csv","$splitDir\test.csv",$tarDst)) {
    "{0,-6} {1}" -f (Test-Path $p), $p
}
Write-Host ("Nested splits\splits folder exists (must be False): " + (Test-Path "$splitDir\splits"))
if (Test-Path $tarDst) { Write-Host ("tar size: {0:N1} MB" -f ((Get-Item $tarDst).Length / 1MB)) }
foreach ($pair in @(@("train.csv",20265),@("val.csv",2534),@("test.csv",2535))) {
    $p = Join-Path $splitDir $pair[0]
    if (Test-Path $p) {
        $n = [System.IO.File]::ReadAllLines($p).Length
        "{0}: {1} lines (expected {2} = rows + header)" -f $pair[0], $n, $pair[1]
    }
}
if (Test-Path "$splitDir\train.csv") { Write-Host ("train.csv header: " + (Get-Content "$splitDir\train.csv" -TotalCount 1)) }
