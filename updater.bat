@echo off
REM Fast-forward Cortex AGI source from GitHub. data/ and trained weights stay.
REM Interpreter is python.env in the project root (PYTHON=path, or a bare path).
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set GIT_TERMINAL_PROMPT=0
set GIT_OPTIONAL_LOCKS=0
if "%AGI_UPDATE_REPO%"=="" set AGI_UPDATE_REPO=codero-sus/agi

if defined AGI_PYTHON set "PYTHON=%AGI_PYTHON%"
if not defined PYTHON (
  if exist python.env (
    call :read_python python.env
  ) else (
    echo updater: put the Python path in python.env at the project root
    exit /b 1
  )
)
if not defined PYTHON (
  echo updater: put the Python path in python.env at the project root
  exit /b 1
)

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

for /f "delims=" %%i in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH=%%i"
if /i "%BRANCH%"=="HEAD" (
  echo updater: detached HEAD — checkout a branch
  exit /b 1
)

echo python: %PYTHON%  ^(python.env^)
echo fetching origin/%BRANCH%…
git fetch --depth 50 origin "%BRANCH%"
if errorlevel 1 (
  echo updater: fetch failed
  exit /b 1
)
for /f "delims=" %%i in ('git rev-parse --short HEAD') do set "BEFORE=%%i"
for /f "delims=" %%i in ('git rev-parse --short origin/%BRANCH%') do set "REMOTE=%%i"
if /i "%BEFORE%"=="%REMOTE%" (
  echo already current ^(%BEFORE%^)
  exit /b 0
)
git merge --ff-only "origin/%BRANCH%"
if errorlevel 1 (
  echo updater: fast-forward failed ^(dirty or diverged^)
  exit /b 1
)
for /f "delims=" %%i in ('git rev-parse --short HEAD') do set "AFTER=%%i"
echo source %BEFORE% → %AFTER%

if exist requirements.txt (
  echo syncing requirements with %PYTHON% …
  "%PYTHON%" -m pip install -r requirements.txt -q --disable-pip-version-check
  if errorlevel 1 echo updater: pip failed ^(source is updated; install deps yourself^)
)

echo Cortex AGI updated. Restart: %PYTHON% -m agi
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
