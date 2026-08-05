@echo off
REM Robust matrix rebuild runner.
REM
REM Two previous attempts died mid-run after several hours because the Python
REM process was piped into Tee-Object; when the parent shell went away the pipe
REM broke and took Python with it. This writes straight to a file with no pipe,
REM so the process owns its own stdout and survives the shell that launched it.
REM
REM Usage:  scripts\rebuild_matrices.bat
REM Log:    rebuild.log  (overwritten each run)

cd /d "%~dp0.."

REM Keep stdout unbuffered so the log reflects real progress.
set PYTHONUNBUFFERED=1
REM Log through UTF-8 so Polish and Turkish club names cannot raise
REM UnicodeEncodeError on a cp1252 console. The previous run logged 190 of them.
set PYTHONIOENCODING=utf-8
REM TensorFlow chatter adds noise without adding information.
set TF_CPP_MIN_LOG_LEVEL=3

echo Rebuild started %DATE% %TIME%
python -m backend.models.training_pipeline --skip-discovery > rebuild.log 2>&1
echo Rebuild finished %DATE% %TIME% with exit code %ERRORLEVEL%
