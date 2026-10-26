param(
    [Parameter(Mandatory=$true)][int]$TrainingProcessId,
    [ValidateSet('training','evidence','application')][string]$Stage = 'training',
    [int]$MaximumMinutes = 0
)
$ErrorActionPreference = 'Stop'
$trainingProcess = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $TrainingProcessId)
$expectedCommand = switch ($Stage) {
    'training' {'*complete_research.py experiments*'}
    'evidence' {'*complete_evidence_pipeline.py --training-pid *'}
    'application' {'*uvicorn app.main:app --host 127.0.0.1 --port 8089*'}
}
if ($Stage -eq 'application' -and ($MaximumMinutes -lt 1 -or $MaximumMinutes -gt 60)) {
    throw 'La revisión de aplicación exige un plazo temporal de 1 a 60 minutos.'
}
if ($null -eq $trainingProcess -or $trainingProcess.CommandLine -notlike $expectedCommand) {
    throw 'El PID no pertenece al estudio local autorizado.'
}
$originalStart = (Get-Process -Id $TrainingProcessId).StartTime
$awakeStarted = Get-Date
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class DentalTrainingPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
try {
    if ([DentalTrainingPower]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        throw 'No se pudo establecer el requisito temporal de ejecución.'
    }
    while ($true) {
        if ($MaximumMinutes -gt 0 -and ((Get-Date) - $awakeStarted).TotalMinutes -ge $MaximumMinutes) { break }
        $activeTraining = Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
        if ($null -eq $activeTraining -or $activeTraining.StartTime -ne $originalStart) { break }
        Start-Sleep -Seconds 20
    }
}
finally {
    # Reset the ordinary idle timer once before releasing the temporary request,
    # so the system does not immediately sleep at the next review step.
    [void][DentalTrainingPower]::SetThreadExecutionState([uint32]1)
    [void][DentalTrainingPower]::SetThreadExecutionState([uint32]2147483648)
}
