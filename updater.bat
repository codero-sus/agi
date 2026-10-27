@echo off
REM Fast-forward Cortex AGI source from GitHub. data/ and trained weights stay.
REM Interpreter: 2PY2, else optional python.env, else python / .venv. (Windows)
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set GIT_TERMINAL_PROMPT=0
set GIT_OPTIONAL_LOCKS=0
if "%AGI_UPDATE_REPO%"=="" set AGI_UPDATE_REPO=codero-sus/agi
if "%AGI_UPDATE_REF%"=="" set AGI_UPDATE_REF=arena/01a0d380-agi

REM 2PY2 starts with a digit — %2PY2% would be %%2. Delayed expansion works.
if defined 2PY2 (
  set "PYTHON=!2PY2!"
  set "SRC=2PY2"
)
if not defined PYTHON (
  if exist python.env (
    call :read_python python.env
    if defined PYTHON set "SRC=python.env"
  )
)
if not defined PYTHON (
  if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
)
if not defined PYTHON set "PYTHON=python"
if not defined SRC set "SRC=default"

where git >nul 2>&1
if errorlevel 1 (
  echo updater: git not installed
  exit /b 1
)
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  echo updater: not a git checkout
  exit /b 1
)

set "ORIGIN="
for /f "delims=" %%i in ('git remote get-url origin 2^>nul') do set "ORIGIN=%%i"
if not defined ORIGIN (
  echo updater: no origin remote
  exit /b 1
)
echo !ORIGIN! | findstr /i /c:"github.com/%AGI_UPDATE_REPO%" /c:"github.com:%AGI_UPDATE_REPO%" >nul
if errorlevel 1 (
  echo updater: origin is not github.com/%AGI_UPDATE_REPO% — refusing
  exit /b 1
)

for /f "delims=" %%i in ('git status --porcelain') do (
  echo updater: working tree dirty — commit or stash first ^(data/ and weights are already ignored^)
  exit /b 1
)

echo python: %PYTHON%  ^(%SRC%^)
echo branch: %AGI_UPDATE_REF%
echo fetching origin/%AGI_UPDATE_REF%…
git fetch --depth 50 origin "%AGI_UPDATE_REF%"
if errorlevel 1 (
  echo updater: fetch failed
  exit /b 1
)
for /f "delims=" %%i in ('git rev-parse --abbrev-ref HEAD') do set "CUR=%%i"
if /i not "%CUR%"=="%AGI_UPDATE_REF%" (
  git checkout "%AGI_UPDATE_REF%" 2>nul
  if errorlevel 1 git checkout -B "%AGI_UPDATE_REF%" "origin/%AGI_UPDATE_REF%"
  if errorlevel 1 (
    echo updater: cannot checkout %AGI_UPDATE_REF%
    exit /b 1
  )
)
for /f "delims=" %%i in ('git rev-parse --short HEAD') do set "BEFORE=%%i"
for /f "delims=" %%i in ('git rev-parse --short origin/%AGI_UPDATE_REF%') do set "REMOTE=%%i"
if /i "%BEFORE%"=="%REMOTE%" (
  echo already current ^(%BEFORE%^) on %AGI_UPDATE_REF%
  exit /b 0
)
git merge --ff-only "origin/%AGI_UPDATE_REF%"
if errorlevel 1 (
  echo updater: fast-forward failed ^(dirty or diverged^)
  exit /b 1
)
for /f "delims=" %%i in ('git rev-parse --short HEAD') do set "AFTER=%%i"
echo source %BEFORE% → %AFTER%  ^(%AGI_UPDATE_REF%^)

if exist requirements.txt (
  echo syncing requirements with %PYTHON% …
  "%PYTHON%" -m pip install -r requirements.txt -q --disable-pip-version-check
  if errorlevel 1 echo updater: pip failed ^(source is updated; install deps yourself^)
)

echo Cortex AGI updated. Restart: %PYTHON% -m agi   or   run.bat
exit /b 0

:read_python
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~1") do (
  if /i "%%A"=="PYTHON" (
    set "_PY=%%B"
  ) else if "%%B"=="" (
    set "_PY=%%A"
  )
  if defined _PY (
    set "_PY=!_PY:"=!"
    if defined _PY set "PYTHON=!_PY!"
  )
)
goto :eof
