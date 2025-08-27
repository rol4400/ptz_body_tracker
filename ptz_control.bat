@echo off
REM PTZ Tracker Windows Service Scripts

echo PTZ Camera Tracking System
echo ===========================
echo.

:menu
echo 1. Start Tracking (daemon mode)
echo 2. Stop Tracking
echo 3. Lock onto Person
echo 4. Go to Preset
echo 5. Check Status
echo 6. Test Body Detection Demo
echo 7. Test Camera Connection
echo 8. Install Dependencies
echo 9. Exit
echo.
set /p choice="Select option (1-7): "

if "%choice%"=="1" goto start_daemon
if "%choice%"=="2" goto stop_tracking
if "%choice%"=="3" goto lock_person
if "%choice%"=="4" goto goto_preset
if "%choice%"=="5" goto check_status
if "%choice%"=="6" goto demo_detection
if "%choice%"=="7" goto test_camera
if "%choice%"=="8" goto install_deps
if "%choice%"=="9" goto exit
goto menu

:start_daemon
echo Starting PTZ Tracker in daemon mode...
python main.py --daemon
pause
goto menu

:stop_tracking
echo Stopping tracking...
curl -X POST http://127.0.0.1:8080/api/stop
pause
goto menu

:lock_person
echo Locking onto primary person...
curl -X POST http://127.0.0.1:8080/api/lock
pause
goto menu

:goto_preset
echo Going to default preset...
curl -X POST http://127.0.0.1:8080/api/preset
pause
goto menu

:check_status
echo Checking status...
curl http://127.0.0.1:8080/api/status
pause
goto menu

:demo_detection
echo Starting body detection demo...
python demo.py
pause
goto menu

:test_camera
echo Testing camera connection...
python camera_test.py
pause
goto menu

:install_deps
echo Installing Python dependencies...
pip install -r requirements.txt
pause
goto menu

:exit
echo Goodbye!
exit /b 0