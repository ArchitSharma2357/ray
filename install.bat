@echo off
setlocal

set "ROOT_DIR=%~dp0"
call "%ROOT_DIR%scripts\install.bat" %*
exit /b %ERRORLEVEL%
