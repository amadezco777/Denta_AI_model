$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repoRoot
$pythonPath = Join-Path $repoRoot '.venv-training\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'No existe .venv-training. Consulta README.md para instalar el entorno.' }
# PowerShell 5.1 can represent informational native stderr as ErrorRecords.
# Preserve the server's actual exit status rather than treating INFO as failure.
$ErrorActionPreference = 'Continue'
& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port 8089
$serverExitCode = $LASTEXITCODE
exit $serverExitCode
