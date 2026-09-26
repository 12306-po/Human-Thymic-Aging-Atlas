$ErrorActionPreference = "Stop"
$atlasRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCandidates = @()
if ($env:ATLAS_PYTHON) { $pythonCandidates += $env:ATLAS_PYTHON }
$pythonCandidates += @("D:\anaconda\python.exe", "python.exe")

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    if (Test-Path -LiteralPath $candidate) {
        $pythonExe = $candidate
        break
    }
    $resolved = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($resolved) {
        $pythonExe = $resolved.Source
        break
    }
}
if (-not $pythonExe) { throw "Python was not found. Set ATLAS_PYTHON to a Python environment with requirements.txt installed." }

& $pythonExe -c "import streamlit, pandas, plotly" 2>$null
if ($LASTEXITCODE -ne 0) { throw "The selected Python environment is missing Streamlit dependencies. Install requirements.txt." }

Push-Location $atlasRoot
try {
    & $pythonExe -m streamlit run "$atlasRoot\app.py"
}
finally {
    Pop-Location
}
