@echo off
REM PTZ Camera Tracker - Windows Management Script

setlocal enabledelayedexpansion

set CONTAINER_NAME=ptz-camera-tracker
set IMAGE_NAME=ptz-tracker
set LOG_DIR=logs

REM Create logs directory
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Function to print status
:print_status
echo [INFO] %~1
echo [INFO] %~1 >> "%LOG_DIR%\install.log"
goto :eof

:print_success
echo [SUCCESS] %~1
echo [SUCCESS] %~1 >> "%LOG_DIR%\install.log"
goto :eof

:print_error
echo [ERROR] %~1
echo [ERROR] %~1 >> "%LOG_DIR%\install.log"
goto :eof

REM Check Docker installation
:check_docker
call :print_status "Checking Docker installation..."

docker --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not installed. Please install Docker Desktop."
    echo Download from: https://www.docker.com/products/docker-desktop
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not running. Please start Docker Desktop."
    exit /b 1
)

call :print_success "Docker is installed and running"
goto :eof

REM Build container
:build_container
call :print_status "Building PTZ Camera Tracker container..."

docker-compose build ptz-tracker
if errorlevel 1 (
    call :print_error "Failed to build container"
    exit /b 1
)

call :print_success "Container built successfully"
goto :eof

REM Start service
:start_service
call :print_status "Starting PTZ Camera Tracker service..."

docker-compose up -d ptz-tracker
if errorlevel 1 (
    call :print_error "Failed to start service"
    exit /b 1
)

call :print_success "Service started successfully"
call :print_status "View logs with: docker-compose logs -f ptz-tracker"
goto :eof

REM Stop service
:stop_service
call :print_status "Stopping PTZ Camera Tracker service..."

docker-compose down
if errorlevel 1 (
    call :print_error "Failed to stop service"
    exit /b 1
)

call :print_success "Service stopped successfully"
goto :eof

REM Show status
:show_status
call :print_status "PTZ Camera Tracker Service Status:"
docker-compose ps

echo.
call :print_status "Resource Usage:"
docker stats "%CONTAINER_NAME%" --no-stream 2>nul || echo Container not running

echo.
call :print_status "Health Status:"
docker inspect "%CONTAINER_NAME%" --format="{{.State.Health.Status}}" 2>nul || echo Container not found
goto :eof

REM Show logs
:show_logs
call :print_status "Showing PTZ Camera Tracker logs..."
docker-compose logs -f ptz-tracker
goto :eof

REM Test OSC
:test_osc
call :print_status "Testing OSC connectivity..."
echo Testing OSC requires python with python-osc package
echo You can test manually by sending OSC messages to port 8081:
echo   /ptz/start - Start tracking
echo   /ptz/stop - Stop tracking
echo   /ptz/relock - Relock to new person
echo   /ptz/status - Get status
goto :eof

REM Show help
:show_help
echo PTZ Camera Tracker - Windows Management Script
echo.
echo Usage: %~n0 [COMMAND]
echo.
echo Commands:
echo   install      - Check dependencies and build container
echo   start        - Start the PTZ tracker service
echo   stop         - Stop the PTZ tracker service
echo   restart      - Restart the PTZ tracker service
echo   status       - Show service status and resource usage
echo   logs         - Show and follow service logs
echo   update       - Update and restart the service
echo   test-osc     - Show OSC testing information
echo   dev          - Start in development mode with GUI
echo   help         - Show this help message
echo.
echo Examples:
echo   %~n0 install       # Initial setup
echo   %~n0 start         # Start service
echo   %~n0 logs          # Monitor logs
goto :eof

REM Main script logic
if "%1"=="install" (
    call :check_docker
    call :build_container
    call :print_success "Installation completed"
    call :print_status "Run '%~n0 start' to start the service"
) else if "%1"=="start" (
    call :check_docker
    call :start_service
) else if "%1"=="stop" (
    call :stop_service
) else if "%1"=="restart" (
    call :stop_service
    timeout /t 2 /nobreak >nul
    call :start_service
) else if "%1"=="status" (
    call :show_status
) else if "%1"=="logs" (
    call :show_logs
) else if "%1"=="update" (
    call :check_docker
    call :build_container
    call :stop_service
    timeout /t 2 /nobreak >nul
    call :start_service
    call :print_success "Service updated successfully"
) else if "%1"=="test-osc" (
    call :test_osc
) else if "%1"=="dev" (
    call :print_status "Starting in development mode..."
    docker-compose --profile dev up ptz-tracker-dev
) else if "%1"=="help" (
    call :show_help
) else if "%1"=="" (
    call :show_help
) else (
    call :print_error "Unknown command: %1"
    echo.
    call :show_help
    exit /b 1
)