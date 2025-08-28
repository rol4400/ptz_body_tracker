# PTZ Camera Tracker - Windows PowerShell Management Script

param(
    [Parameter(Position=0)]
    [ValidateSet('install', 'start', 'stop', 'restart', 'status', 'logs', 'update', 'test-osc', 'dev', 'help')]
    [string]$Command = 'help'
)

$ContainerName = "ptz-camera-tracker"
$ImageName = "ptz-tracker"
$LogDir = "logs"

# Create logs directory
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

function Write-Status {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [INFO] $Message"
    Write-Host "[INFO] $Message" -ForegroundColor Blue
    Add-Content -Path "$LogDir\install.log" -Value $logMessage
}

function Write-Success {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [SUCCESS] $Message"
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
    Add-Content -Path "$LogDir\install.log" -Value $logMessage
}

function Write-Error {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [ERROR] $Message"
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    Add-Content -Path "$LogDir\install.log" -Value $logMessage
}

function Test-Docker {
    Write-Status "Checking Docker installation..."
    
    try {
        $dockerVersion = docker --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Docker is not installed. Please install Docker Desktop."
            Write-Host "Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
            return $false
        }
        
        docker info 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Docker is not running. Please start Docker Desktop."
            return $false
        }
        
        Write-Success "Docker is installed and running"
        return $true
    }
    catch {
        Write-Error "Failed to check Docker: $($_.Exception.Message)"
        return $false
    }
}

function Build-Container {
    Write-Status "Building PTZ Camera Tracker container..."
    
    try {
        docker-compose build ptz-tracker
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to build container"
            return $false
        }
        
        Write-Success "Container built successfully"
        return $true
    }
    catch {
        Write-Error "Failed to build container: $($_.Exception.Message)"
        return $false
    }
}

function Start-Service {
    Write-Status "Starting PTZ Camera Tracker service..."
    
    try {
        docker-compose up -d ptz-tracker
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to start service"
            return $false
        }
        
        Write-Success "Service started successfully"
        Write-Status "View logs with: docker-compose logs -f ptz-tracker"
        return $true
    }
    catch {
        Write-Error "Failed to start service: $($_.Exception.Message)"
        return $false
    }
}

function Stop-Service {
    Write-Status "Stopping PTZ Camera Tracker service..."
    
    try {
        docker-compose down
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to stop service"
            return $false
        }
        
        Write-Success "Service stopped successfully"
        return $true
    }
    catch {
        Write-Error "Failed to stop service: $($_.Exception.Message)"
        return $false
    }
}

function Show-Status {
    Write-Status "PTZ Camera Tracker Service Status:"
    docker-compose ps
    
    Write-Host ""
    Write-Status "Resource Usage:"
    try {
        docker stats $ContainerName --no-stream 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Container not running" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "Container not running" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Status "Health Status:"
    try {
        $healthStatus = docker inspect $ContainerName --format="{{.State.Health.Status}}" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Health: $healthStatus" -ForegroundColor $(if ($healthStatus -eq "healthy") { "Green" } else { "Yellow" })
        } else {
            Write-Host "Container not found" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "Container not found" -ForegroundColor Yellow
    }
}

function Show-Logs {
    Write-Status "Showing PTZ Camera Tracker logs..."
    docker-compose logs -f ptz-tracker
}

function Test-OSC {
    Write-Status "Testing OSC connectivity..."
    Write-Host ""
    Write-Host "OSC Testing Information:" -ForegroundColor Cyan
    Write-Host "Default OSC port: 8081" -ForegroundColor White
    Write-Host ""
    Write-Host "Available OSC commands:" -ForegroundColor Yellow
    Write-Host "  /ptz/start    - Start tracking" -ForegroundColor White
    Write-Host "  /ptz/stop     - Stop tracking" -ForegroundColor White
    Write-Host "  /ptz/relock   - Relock to new person" -ForegroundColor White
    Write-Host "  /ptz/status   - Get status" -ForegroundColor White
    Write-Host ""
    Write-Host "Test with Python:" -ForegroundColor Cyan
    Write-Host "pip install python-osc" -ForegroundColor Gray
    Write-Host "python -c 'from pythonosc import udp_client; client = udp_client.SimpleUDPClient(`"localhost`", 8081); client.send_message(`"/ptz/start`", None)'" -ForegroundColor Gray
}

function Start-Dev {
    Write-Status "Starting in development mode..."
    docker-compose --profile dev up ptz-tracker-dev
}

function Show-Help {
    Write-Host "PTZ Camera Tracker - Windows PowerShell Management Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\manage.ps1 [COMMAND]" -ForegroundColor White
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  install      - Check dependencies and build container" -ForegroundColor White
    Write-Host "  start        - Start the PTZ tracker service" -ForegroundColor White
    Write-Host "  stop         - Stop the PTZ tracker service" -ForegroundColor White
    Write-Host "  restart      - Restart the PTZ tracker service" -ForegroundColor White
    Write-Host "  status       - Show service status and resource usage" -ForegroundColor White
    Write-Host "  logs         - Show and follow service logs" -ForegroundColor White
    Write-Host "  update       - Update and restart the service" -ForegroundColor White
    Write-Host "  test-osc     - Show OSC testing information" -ForegroundColor White
    Write-Host "  dev          - Start in development mode with GUI" -ForegroundColor White
    Write-Host "  help         - Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\manage.ps1 install       # Initial setup" -ForegroundColor Gray
    Write-Host "  .\manage.ps1 start         # Start service" -ForegroundColor Gray
    Write-Host "  .\manage.ps1 logs          # Monitor logs" -ForegroundColor Gray
}

# Main script logic
switch ($Command) {
    'install' {
        if (Test-Docker) {
            if (Build-Container) {
                Write-Success "Installation completed"
                Write-Status "Run '.\manage.ps1 start' to start the service"
            }
        }
    }
    'start' {
        if (Test-Docker) {
            Start-Service
        }
    }
    'stop' {
        Stop-Service
    }
    'restart' {
        if (Stop-Service) {
            Start-Sleep -Seconds 2
            if (Test-Docker) {
                Start-Service
            }
        }
    }
    'status' {
        Show-Status
    }
    'logs' {
        Show-Logs
    }
    'update' {
        if (Test-Docker) {
            if (Build-Container) {
                if (Stop-Service) {
                    Start-Sleep -Seconds 2
                    if (Start-Service) {
                        Write-Success "Service updated successfully"
                    }
                }
            }
        }
    }
    'test-osc' {
        Test-OSC
    }
    'dev' {
        Start-Dev
    }
    'help' {
        Show-Help
    }
    default {
        Write-Error "Unknown command: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
}