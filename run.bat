@echo off
setlocal

set "ROOT_DIR=%~dp0"
call "%ROOT_DIR%scripts\run.bat" %*
exit /b %ERRORLEVEL%
