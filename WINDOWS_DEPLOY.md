# PTZ Camera Tracker - Quick Setup and Control

This Windows deployment provides a complete containerized PTZ camera tracking solution with OSC remote control.

## Quick Start

### 1. Install and Start
```powershell
# PowerShell (Recommended)
.\manage.ps1 install
.\manage.ps1 start

# Or using Batch
manage.bat install
manage.bat start
```

### 2. Monitor Status
```powershell
.\manage.ps1 status
.\manage.ps1 logs
```

### 3. OSC Remote Control
The system provides OSC control on port 8081:
- `/ptz/start` - Start tracking
- `/ptz/stop` - Stop tracking  
- `/ptz/relock` - Relock to new person
- `/ptz/status` - Get status

## Management Commands

### PowerShell Script (manage.ps1)
```powershell
.\manage.ps1 install      # Initial setup
.\manage.ps1 start        # Start service
.\manage.ps1 stop         # Stop service
.\manage.ps1 restart      # Restart service
.\manage.ps1 status       # Show status
.\manage.ps1 logs         # View logs
.\manage.ps1 update       # Update and restart
.\manage.ps1 test-osc     # OSC testing info
.\manage.ps1 dev          # Development mode
```

### Batch Script (manage.bat)
```batch
manage.bat install       # Initial setup
manage.bat start         # Start service
manage.bat stop          # Stop service
manage.bat restart       # Restart service
manage.bat status        # Show status
manage.bat logs          # View logs
manage.bat update        # Update and restart
manage.bat test-osc      # OSC testing info
manage.bat dev           # Development mode
```

## Configuration

### Camera Settings
Edit `docker-compose.yml` environment variables:
```yaml
environment:
  - CAMERA_IP=192.168.0.251
  - CAMERA_PORT=5678
  - OSC_PORT=8081
  - REST_PORT=8080
```

### Resource Limits
Production service is configured with:
- CPU Limit: 2 cores
- Memory Limit: 1GB
- Health checks every 30s

## Troubleshooting

### Check Docker Status
```powershell
docker --version
docker info
```

### View Container Logs
```powershell
.\manage.ps1 logs
# Or directly:
docker-compose logs -f ptz-tracker
```

### Service Status
```powershell
.\manage.ps1 status
# Shows: container status, resource usage, health status
```

### Development Mode
```powershell
.\manage.ps1 dev
# Runs with GUI for testing
```

## Integration Examples

### OSC Control with Python
```python
from pythonosc import udp_client

client = udp_client.SimpleUDPClient("localhost", 8081)
client.send_message("/ptz/start", None)
```

### REST API
```powershell
# Start tracking
Invoke-RestMethod -Uri "http://localhost:8080/start" -Method POST

# Stop tracking  
Invoke-RestMethod -Uri "http://localhost:8080/stop" -Method POST

# Get status
Invoke-RestMethod -Uri "http://localhost:8080/status" -Method GET
```

## Production Deployment

### Background Service
```powershell
.\manage.ps1 install
.\manage.ps1 start
# Service runs in background with automatic restart
```

### System Monitoring
```powershell
# Check if running
docker ps | findstr ptz-tracker

# Resource usage
docker stats ptz-camera-tracker --no-stream

# Health status
docker inspect ptz-camera-tracker --format="{{.State.Health.Status}}"
```

## Logs and Debugging

### Log Locations
- Container logs: `docker-compose logs ptz-tracker`
- Installation logs: `logs/install.log`
- Application logs: Available through Docker logging

### Common Issues
1. **Docker not running**: Start Docker Desktop
2. **Port conflicts**: Check ports 8080, 8081 are available
3. **Camera connection**: Verify camera IP (192.168.0.251) and port (5678)
4. **Permission issues**: Run PowerShell as Administrator if needed

### Support
- View help: `.\manage.ps1 help`
- Test OSC: `.\manage.ps1 test-osc`
- Development mode: `.\manage.ps1 dev`