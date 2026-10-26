param([int]$Port = 8080, [switch]$Background, [ValidateSet('wsl','native')][string]$Backend = 'wsl')
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if ($Backend -eq 'wsl') {
    if ($Port -ne 8080) { throw 'El perfil WSL usa el puerto 8080.' }
    $linuxPathOutput = & wsl -d Ubuntu -- wslpath -a $projectRoot.Replace('\','/')
    if ($LASTEXITCODE -ne 0 -or -not $linuxPathOutput) { throw 'No se pudo resolver el proyecto en WSL Ubuntu.' }
    $linuxRoot = $linuxPathOutput.Trim()
    $wslArgs = @('-d','Ubuntu','--','bash',"$linuxRoot/scripts/labeling/start_labelstudio_wsl.sh")
    if ($Background) {
        try {
            $health = Invoke-WebRequest -Uri 'http://127.0.0.1:8080/health' -UseBasicParsing -TimeoutSec 2
            if ($health.StatusCode -eq 200) { Write-Output 'Label Studio ya responde en http://127.0.0.1:8080'; return }
        } catch { }
        $logRoot = Join-Path $projectRoot 'data/labelstudio/logs'
        New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
        $wslArgumentLine = '-d Ubuntu -- bash "' + $linuxRoot + '/scripts/labeling/start_labelstudio_wsl.sh"'
        $process = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32/wsl.exe') -ArgumentList $wslArgumentLine -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logRoot 'wsl.stdout.log') -RedirectStandardError (Join-Path $logRoot 'wsl.stderr.log') -PassThru
        $process.Id | Set-Content -LiteralPath (Join-Path $logRoot 'wsl.pid')
        Write-Output "Label Studio en WSL, proceso $($process.Id): http://127.0.0.1:8080"
    } else {
        & wsl @wslArgs
    }
    return
}
$lsPython = Join-Path $projectRoot '.venv-labelstudio/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $lsPython)) {
    throw 'Instala requirements-labelstudio.txt en .venv-labelstudio antes de iniciar.'
}
$lsData = Join-Path $projectRoot 'data/labelstudio/state'
$lsMedia = Join-Path $projectRoot 'data/labelstudio/media'
New-Item -ItemType Directory -Force -Path $lsData,$lsMedia | Out-Null
# DEBUG=release se hereda de algunos hosts; Django necesita un booleano.
$env:DEBUG = 'false'
$env:LABEL_STUDIO_BASE_DATA_DIR = $lsData
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = 'true'
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $lsMedia
$env:LABEL_STUDIO_COLLECT_ANALYTICS = 'false'
$env:LABEL_STUDIO_SENTRY_DSN = ''
$env:LABEL_STUDIO_FRONTEND_SENTRY_DSN = ''
$env:DJANGO_HOSTNAME = "http://127.0.0.1:$Port"
$env:LABEL_STUDIO_HOST = "http://127.0.0.1:$Port"
$lsArgs = @('-m','label_studio.server','start','--no-browser','--internal-host','127.0.0.1','--host',"http://127.0.0.1:$Port",'--port',"$Port",'--data-dir',$lsData)
if ($Background) {
    $stdout = Join-Path $lsData 'server.stdout.log'
    $stderr = Join-Path $lsData 'server.stderr.log'
    $quotedArgs = $lsArgs | ForEach-Object { '"' + $_ + '"' }
    $process = Start-Process -FilePath $lsPython -ArgumentList $quotedArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $process.Id | Set-Content -LiteralPath (Join-Path $lsData 'server.pid')
    Write-Output "Label Studio PID $($process.Id): http://127.0.0.1:$Port"
} else {
    & $lsPython @lsArgs
}
