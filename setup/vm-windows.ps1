# Prepare a Windows machine with Intel hardware (the Lunar Lake cloud VM) to run afterpass captures.
# Run from a clone of the repo:  powershell -ExecutionPolicy Bypass -File setup\vm-windows.ps1
# Safe to rerun: it installs uv if missing, syncs the capture environment and lists the OpenVINO devices.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    # the official installer puts uv in %USERPROFILE%\.local\bin
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv --version

Set-Location "$repo\capture"
uv sync --group dev
uv run python -c "import openvino as ov; c = ov.Core(); print('openvino', ov.get_version()); [print(d, c.get_property(d, 'FULL_DEVICE_NAME')) for d in c.available_devices]"
