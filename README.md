# PTZ Camera Tracking System

A real-time body tracking system for PTZ cameras with ONVIF support, designed for stage performance tracking.

## Features

- Real-time body tracking using OpenCV HOG detector
- ONVIF PTZ camera control
- Multiple person detection and tracking
- Smooth pan movements with dead zone
- Preset management for lost tracking
- OSC and REST API control
- Bitfocus Companion integration
- Low latency, lightweight operation

## Requirements

- Windows 10/11
- Python 3.8+ (tested with Python 3.13)
- PTZ camera with ONVIF support
- Network connection to camera (192.168.0.251)

## Installation

1. Clone or download this repository
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

Edit `config.json` to set your camera IP, credentials, and tracking parameters:

```json
{
  "camera": {
    "ip": "192.168.0.251",
    "username": "admin",
    "password": "admin",
    "stream_url": "rtsp://192.168.0.251:554/stream1"
  }
}
```

## Usage

### Quick Start (Windows)
- **Batch Script**: Run `ptz_control.bat` for a simple menu interface
- **PowerShell**: Run `ptz_control.ps1` for an enhanced interface

### Command Line
```bash
# Start tracking daemon with API/OSC servers
python main.py --daemon

# Quick commands
python main.py --start     # Start tracking
python main.py --stop      # Stop tracking  
python main.py --lock      # Lock onto primary person
python main.py --preset    # Go to default preset
```

### OSC Commands (Port 8081)
- `/ptz/start` - Start tracking
- `/ptz/stop` - Stop tracking
- `/ptz/lock` - Lock onto primary person
- `/ptz/preset` - Go to default preset
- `/ptz/pan <angle>` - Pan to specific angle

### REST API (Port 8080)
- `POST /api/start` - Start tracking
- `POST /api/stop` - Stop tracking
- `POST /api/lock` - Lock onto primary person
- `POST /api/preset` - Go to default preset
- `GET /api/status` - Get current status

Example API usage:
```bash
curl -X POST http://localhost:8080/api/start
curl -X POST http://localhost:8080/api/lock
curl http://localhost:8080/api/status
```

## Bitfocus Companion Integration

The system provides HTTP REST API endpoints that can be used directly with Bitfocus Companion:

1. **HTTP Request Actions**: Use the REST API endpoints
2. **OSC Actions**: Send OSC messages to port 8081
3. **Example Button Config**:
   - Button 1: `POST http://localhost:8080/api/start` (Start Tracking)
   - Button 2: `POST http://localhost:8080/api/lock` (Lock Person)
   - Button 3: `POST http://localhost:8080/api/preset` (Go Home)

## Technical Notes

- **Body Detection**: Uses OpenCV HOG detector for Python 3.13 compatibility
- **Tracking Algorithm**: Multi-person tracking with confidence scoring
- **Movement Smoothing**: Configurable smoothing and dead zone
- **Error Handling**: Automatic fallback to preset on lost tracking
- **Performance**: Optimized for low latency and resource usage

## Troubleshooting

1. **Camera Connection Issues**: 
   - Verify camera IP and credentials in `config.json`
   - Test ONVIF service at `http://[camera_ip]/onvif/device_service`

2. **No Person Detection**:
   - Ensure adequate lighting
   - Check camera angle and person size
   - Adjust `confidence_threshold` in config

3. **Erratic Movement**:
   - Increase `smoothing_factor` for smoother movement
   - Adjust `dead_zone_width` to reduce sensitivity

## Quick Testing

### Test Body Detection (No Camera Required)
```bash
python demo.py
```
This opens a webcam window showing real-time person detection and tracking. Great for testing the core tracking logic.

### Test Camera Connection
```bash
python camera_test.py
```
Diagnostic tool to test connectivity to your PTZ camera before running the main system.

### Full System Test
```bash
python test_system.py
```

## License

MIT License