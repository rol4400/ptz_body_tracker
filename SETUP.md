# PTZ Camera Tracking System - Setup Guide

## Quick Setup Instructions

### 1. Configure Your Camera
Edit `config.json` with your camera settings:

```json
{
  "camera": {
    "ip": "192.168.0.251",        // Your camera's IP address
    "port": 80,                   // ONVIF port (usually 80)
    "username": "admin",          // Camera username
    "password": "admin",          // Camera password
    "stream_url": "rtsp://192.168.0.251:554/stream1"  // RTSP stream URL
  }
}
```

### 2. Start the System

**Option A: Windows Control Scripts**
- Run `ptz_control.bat` (simple menu)
- Run `ptz_control.ps1` (advanced PowerShell menu)

**Option B: Command Line**
```bash
python main.py --daemon --debug
```

### 3. Control via Bitfocus Companion

**HTTP Request Buttons:**
- Start Tracking: `POST http://localhost:8080/api/start`
- Stop Tracking: `POST http://localhost:8080/api/stop`  
- Lock Person: `POST http://localhost:8080/api/lock`
- Go Home: `POST http://localhost:8080/api/preset`
- Get Status: `GET http://localhost:8080/api/status`

**OSC Buttons (port 8081):**
- Start: `/ptz/start`
- Stop: `/ptz/stop`
- Lock: `/ptz/lock` 
- Home: `/ptz/preset`

## Configuration Options

### Tracking Settings
```json
{
  "tracking": {
    "confidence_threshold": 0.7,    // Minimum detection confidence
    "dead_zone_width": 0.3,         // Center area that doesn't trigger movement
    "smoothing_factor": 0.15,       // Movement smoothing (0.0-1.0)
    "lost_tracking_timeout": 3.0,   // Seconds before going to preset
    "min_detection_size": 0.1       // Minimum person size to track
  }
}
```

### PTZ Settings
```json
{
  "ptz": {
    "pan_speed": 0.3,              // Movement speed (0.0-1.0)
    "max_pan_angle": 180,          // Maximum pan angle
    "min_pan_angle": -180,         // Minimum pan angle  
    "default_preset": 1            // Preset number for lost tracking
  }
}
```

## Troubleshooting

### Camera Connection Issues
1. Verify camera IP address is correct
2. Check username/password
3. Test ONVIF service: `http://[camera_ip]/onvif/device_service`
4. Ensure camera supports ONVIF PTZ control

### Poor Person Detection
1. Ensure good lighting conditions
2. Person should be visible from waist up
3. Adjust `confidence_threshold` (lower = more sensitive)
4. Check camera angle and distance

### Jerky Camera Movement
1. Increase `smoothing_factor` (0.3-0.5 for very smooth)
2. Increase `dead_zone_width` to reduce sensitivity
3. Reduce `pan_speed` for slower movements

### API Not Responding
1. Check if daemon is running: `curl http://localhost:8080/api/status`
2. Verify ports are not blocked by firewall
3. Check log file `ptz_tracker.log` for errors

## Advanced Usage

### Custom Presets
Set up presets on your camera and reference them in config:
```json
{
  "ptz": {
    "default_preset": 2,    // Use preset 2 as home position
    "home_position": {      // Or set specific coordinates
      "pan": 0,
      "tilt": 0,
      "zoom": 0
    }
  }
}
```

### Multiple People Handling
The system automatically selects the "best" person to track based on:
- Size (larger people preferred)
- Confidence level
- Position (center of frame preferred)
- Stability (less movement preferred)

Use the "lock" command to manually select the current primary person.

### Integration with Other Systems
- **OBS Studio**: Use OSC plugin to trigger scene changes
- **QLab**: Send OSC commands from cues
- **Touch OSC**: Create mobile control interface
- **Node-RED**: Build complex automation workflows

## System Requirements

**Minimum:**
- Windows 10
- Python 3.8+
- 4GB RAM
- Network connection to camera

**Recommended:**
- Windows 11
- Python 3.10+
- 8GB RAM
- Dedicated network for camera traffic
- SSD storage for better performance