@echo off
REM Launch Cortex AGI (Windows).
REM Interpreter: 2PY2, else optional python.env, else python / .venv.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM 2PY2 starts with a digit — %2PY2% would be %%2. Delayed expansion works.
if defined 2PY2 (
  set "PY=!2PY2!"
  set "SRC=2PY2"
)
if not defined PY (
  if exist python.env (
    call :read_python python.env
    if defined PY set "SRC=python.env"
  )
)
if not defined PY (
  if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
)
if not defined PY set "PY=python"
if not defined SRC set "SRC=default"

echo python: %PY%  ^(%SRC%^)
"%PY%" run.py %*
exit /b %ERRORLEVEL%

:read_python
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~1") do (
  if /i "%%A"=="PYTHON" (
    set "_PY=%%B"
  ) else if "%%B"=="" (
    set "_PY=%%A"
  )
  if defined _PY (
    set "_PY=!_PY:"=!"
    if defined _PY set "PY=!_PY!"
  )
)
goto :eof
