@echo off
title UROP Step 4 - MVTec benchmark
cd /d "%~dp0"
set "PY=%USERPROFILE%\UROP_data\venv\Scripts\python.exe"
if not exist "outputs\logs" mkdir "outputs\logs"
echo [%date% %time%] Step 4 started > outputs\logs\step4_status.txt
echo === 1/2 Downloading MVTec AD transistor + cable (about 0.9 GB) - progress goes to outputs\logs\setup_mvtec.log ===
"%PY%" -u setup_data.py mvtec > outputs\logs\setup_mvtec.log 2>&1
set ERR=%errorlevel%
type outputs\logs\setup_mvtec.log
if not "%ERR%"=="0" goto fail
echo [%date% %time%] download done >> outputs\logs\step4_status.txt
echo === 2/2 Running the benchmark (about 20-40 min) ===
"%PY%" -u run_all.py bench
if errorlevel 1 goto fail
echo [%date% %time%] ALL DONE >> outputs\logs\step4_status.txt
echo === ALL DONE - you can close this window ===
goto end
:fail
echo [%date% %time%] FAILED >> outputs\logs\step4_status.txt
echo !!! Something failed - see the messages above
:end
pause
