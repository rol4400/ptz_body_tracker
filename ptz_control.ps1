# PTZ Tracker PowerShell Control Script

function Show-Menu {
    Write-Host "PTZ Camera Tracking System" -ForegroundColor Green
    Write-Host "===========================" -ForegroundColor Green
    Write-Host ""
    Write-Host "1. Start Tracking (daemon mode)"
    Write-Host "2. Stop Tracking" 
    Write-Host "3. Lock onto Person"
    Write-Host "4. Go to Preset"
    Write-Host "5. Check Status"
    Write-Host "6. Body Detection Demo"
    Write-Host "7. Test Camera Connection"
    Write-Host "8. Install Dependencies"
    Write-Host "9. View Logs"
    Write-Host "10. Exit"
    Write-Host ""
}

function Start-PTZDaemon {
    Write-Host "Starting PTZ Tracker in daemon mode..." -ForegroundColor Yellow
    python main.py --daemon --debug
}

function Stop-PTZTracking {
    Write-Host "Stopping tracking..." -ForegroundColor Yellow
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/stop" -Method Post
        Write-Host "Response: $($response.message)" -ForegroundColor Green
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
    }
}

function Lock-PTZPerson {
    Write-Host "Locking onto primary person..." -ForegroundColor Yellow
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/lock" -Method Post
        Write-Host "Response: $($response.message)" -ForegroundColor Green
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
    }
}

function Goto-PTZPreset {
    Write-Host "Going to default preset..." -ForegroundColor Yellow
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/preset" -Method Post
        Write-Host "Response: $($response.message)" -ForegroundColor Green
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
    }
}

function Get-PTZStatus {
    Write-Host "Checking status..." -ForegroundColor Yellow
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/status" -Method Get
        Write-Host "Status:" -ForegroundColor Green
        $response.data | Format-List
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
    }
}

function Install-PTZDependencies {
    Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Dependencies installed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Error installing dependencies!" -ForegroundColor Red
    }
}

function Show-PTZDemo {
    Write-Host "Starting body detection demo..." -ForegroundColor Yellow
    python demo.py
}

function Test-PTZCamera {
    Write-Host "Testing camera connection..." -ForegroundColor Yellow
    python camera_test.py
}

function Show-PTZLogs {
    Write-Host "Recent log entries:" -ForegroundColor Yellow
    if (Test-Path "ptz_tracker.log") {
        Get-Content "ptz_tracker.log" -Tail 20
    } else {
        Write-Host "Log file not found." -ForegroundColor Red
    }
}

# Main menu loop
do {
    Show-Menu
    $choice = Read-Host "Select option (1-10)"
    
    switch ($choice) {
        "1" { Start-PTZDaemon; break }
        "2" { Stop-PTZTracking }
        "3" { Lock-PTZPerson }
        "4" { Goto-PTZPreset }
        "5" { Get-PTZStatus }
        "6" { Show-PTZDemo }
        "7" { Test-PTZCamera }
        "8" { Install-PTZDependencies }
        "9" { Show-PTZLogs }
        "10" { Write-Host "Goodbye!" -ForegroundColor Green; break }
        default { Write-Host "Invalid option!" -ForegroundColor Red }
    }
    
    if ($choice -ne "1" -and $choice -ne "10") {
        Write-Host ""
        Read-Host "Press Enter to continue"
    }
} while ($choice -ne "10" -and $choice -ne "1")