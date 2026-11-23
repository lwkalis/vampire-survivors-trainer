param(
    [switch]$WithDeepLearning
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$Python = Get-Command py -ErrorAction SilentlyContinue
if ($Python) {
    & py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv. Install Python 3 and retry.' }
    $VenvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
} else {
    $Python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $Python) { throw 'Python 3 was not found. Install Python and enable Add Python to PATH.' }
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv. Install Python 3 and retry.' }
    $VenvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip.' }
& $VenvPython -m pip install -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install Windows dependencies.' }

if ($WithDeepLearning) {
    & $VenvPython -m pip install torch
    if ($LASTEXITCODE -ne 0) { throw 'Could not install PyTorch.' }
}

& $VenvPython -B -c "import cv2, mss, numpy; print('Windows dependencies OK')"
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
if ($WithDeepLearning) {
    & $VenvPython -B -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch check failed.' }
}

Write-Host ''
Write-Host 'Setup complete. Start the game, leave it visible, then run:'
if ($WithDeepLearning) {
    Write-Host '.\.venv\Scripts\python.exe -B -u .\train_agent.py --full-run --deep-learning'
} else {
    Write-Host '.\.venv\Scripts\python.exe -B -u .\train_agent.py --full-run'
    Write-Host 'Run setup_windows.ps1 -WithDeepLearning to install PyTorch for DQN mode.'
}

