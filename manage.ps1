# PTZ Camera Tracker Management Script
param([string]$Command = "help")

$ContainerName = "ptz-camera-tracker"

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Blue }
function Write-OK($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Err($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

switch ($Command) {
    "install" {
        Write-Info "Building PTZ Camera Tracker container..."
        docker-compose build ptz-tracker
        if ($LASTEXITCODE -eq 0) { Write-OK "Container built successfully" }
        else { Write-Err "Build failed" }
    }
    "start" {
        Write-Info "Starting PTZ Camera Tracker service..."
        docker-compose up -d ptz-tracker
        if ($LASTEXITCODE -eq 0) { Write-OK "Service started" }
        else { Write-Err "Start failed" }
    }
    "stop" {
        Write-Info "Stopping PTZ Camera Tracker service..."
        docker-compose down
        if ($LASTEXITCODE -eq 0) { Write-OK "Service stopped" }
        else { Write-Err "Stop failed" }
    }
    "status" {
        Write-Info "Service Status:"
        docker-compose ps
        Write-Host ""
        docker stats $ContainerName --no-stream 2>$null
    }
    "logs" {
        docker-compose logs -f ptz-tracker
    }
    "test-gpu" {
        Write-Info "Testing GPU support..."
        docker run --rm --gpus all nvidia/cuda:13.0-runtime-ubuntu22.04 nvidia-smi
    }
    "help" {
        Write-Host "PTZ Camera Tracker Management" -ForegroundColor Cyan
        Write-Host "Usage: .\manage.ps1 [command]" -ForegroundColor White
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  install   - Build container"
        Write-Host "  start     - Start service"
        Write-Host "  stop      - Stop service"
        Write-Host "  status    - Show status"
        Write-Host "  logs      - Show logs"
        Write-Host "  test-gpu  - Test GPU support"
        Write-Host "  help      - Show this help"
    }
    default {
        Write-Err "Unknown command: $Command"
        .\manage.ps1 help
    }
}
