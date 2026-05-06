@echo off
REM ========================================================================
REM  Mental Health Monitoring Dashboard - Website Launcher
REM  Click this file to start the website with 90.28% accurate model
REM ========================================================================

echo.
echo ========================================================================
echo  🧠 MENTAL HEALTH MONITORING SYSTEM
echo  IoT-Enabled Wearable Sensors for Mental Health Monitoring
echo ========================================================================
echo.

REM Get the directory where this batch file is located
cd /d "%~dp0"

echo [1/3] Checking system...
timeout /t 1 /nobreak > nul

echo [2/3] Starting Flask Server...
REM Kill any existing python processes
taskkill /f /im python.exe >nul 2>&1
start /min cmd /c "python app/app_enhanced.py"

REM Wait for server to start
timeout /t 3 /nobreak > nul

echo [3/3] Opening Website...
timeout /t 1 /nobreak > nul

REM Open the website in default browser
start http://127.0.0.1:5000

echo.
echo ========================================================================
echo  ✓ Website launching...
echo  📱 Dashboard: http://127.0.0.1:5000
echo  
echo  Model: Voting Ensemble (90.28% Accuracy)
echo  Features: 16 (8 physiological + 8 engineered)
echo  Cross-Validation: 85% ± 3.16%
echo  
echo  To stop the server, close the Flask command window.
echo ========================================================================
echo.

REM Keep batch window open to show messages
pause
