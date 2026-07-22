$ErrorActionPreference = "Stop"

$ProjectDir = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectDir

$PythonCandidates = @()
if ($env:PHISHGUARD_BUILD_PYTHON) {
    $PythonCandidates += $env:PHISHGUARD_BUILD_PYTHON
}
$PythonCandidates += @(
    (Join-Path $ProjectDir ".venv\Scripts\python.exe"),
    (Join-Path $ProjectDir "..\.venv\Scripts\python.exe"),
    "python"
)

$Python = $null
foreach ($Candidate in $PythonCandidates) {
    try {
        & $Candidate --version | Out-Null
        $Python = $Candidate
        break
    } catch {
        continue
    }
}

if (-not $Python) {
    throw "Python was not found. Install Python 3.10+ or set PHISHGUARD_BUILD_PYTHON."
}

Write-Host "Using Python: $Python"

& $Python -m pip install -r requirements.txt
& $Python -m pip install "pyinstaller>=6,<7"

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "backend-build", "backend-dist"

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --workpath "backend-build" `
    --distpath "backend-dist" `
    "phishguard-backend.spec"

$BackendExe = Join-Path $ProjectDir "backend-dist\phishguard-backend\phishguard-backend.exe"
if (-not (Test-Path $BackendExe)) {
    throw "Backend build failed: $BackendExe was not created."
}

Write-Host "Backend ready: $BackendExe"
