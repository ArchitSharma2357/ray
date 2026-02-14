@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"

if defined PYTHON_BIN (
  "%PYTHON_BIN%" "%ROOT_DIR%\scripts\orion_portable.py" install %*
  exit /b %ERRORLEVEL%
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  py -3 "%ROOT_DIR%\scripts\orion_portable.py" install %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  python "%ROOT_DIR%\scripts\orion_portable.py" install %*
  exit /b %ERRORLEVEL%
)

echo [install.bat] Python 3.9+ was not found in PATH.
exit /b 1
