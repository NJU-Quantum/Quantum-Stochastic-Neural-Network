param(
    [int]$CalibrationProcessId = 0,
    [switch]$SkipCalibration,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Output = Join-Path $Root "outputs\qgan\random_local_rotated_werner"
$Runner = "scripts\run_rotated_werner_qgan.py"

function Invoke-ExperimentStage {
    param([Parameter(Mandatory = $true)][string]$Stage)

    $stdout = Join-Path $Output ("{0}_full_stdout.log" -f $Stage)
    $stderr = Join-Path $Output ("{0}_full_stderr.log" -f $Stage)
    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList @("-u", $Runner, "--stage", $Stage, "--device", "cpu") `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Stage '$Stage' failed with exit code $($process.ExitCode). See $stderr"
    }
}

if (-not $SkipCalibration) {
    if ($CalibrationProcessId -le 0) {
        throw "CalibrationProcessId is required unless -SkipCalibration is supplied."
    }
    $calibration = Get-Process -Id $CalibrationProcessId -ErrorAction Stop
    $calibration.WaitForExit()
    # A process obtained by PID is not a child of this PowerShell process, so
    # ExitCode can be unavailable after it exits.  Validate durable artifacts
    # and stderr instead of treating a missing ExitCode as a failure.
    $calibrationError = Join-Path $Output "calibration_full_stderr.log"
    $selection = Join-Path $Output "calibration\selection.json"
    if ((Test-Path $calibrationError) -and (Get-Item $calibrationError).Length -gt 0) {
        throw "Calibration wrote an error log. See $calibrationError"
    }
    if (-not (Test-Path $selection)) {
        throw "Calibration selection is missing: $selection"
    }
}

Invoke-ExperimentStage -Stage "validation"
Invoke-ExperimentStage -Stage "formal"
