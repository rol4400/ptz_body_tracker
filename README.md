# PTZ Camera Tracking System

A real-time body tracking system for PTZ cameras with ONVIF support, designed for stage performance tracking.

## Features

- Real-time body tracking using YOLOv8 with GPU acceleration
- ONVIF PTZ camera control with VISCA protocol support
- Multiple person detection and tracking with ID persistence
- Smooth pan movements with configurable dead zone
- Preset management for lost tracking recovery
- OSC control interface for real-time control
- Bitfocus Companion integration ready
- Docker containerization with GPU support
- Low latency, lightweight operation

## Requirements

- **Hardware**: NVIDIA GPU (recommended for best performance)
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.11+ (or use Docker)
- **Camera**: PTZ camera with ONVIF/VISCA support
- **Network**: Ethernet connection to camera

## Quick Start

### Using Docker (Recommended)
```bash
# Clone repository
git clone <repository-url>
cd ptz_body_tracker

# Configure your camera settings
# Edit config.json with your camera IP and credentials

# Start with Docker Compose
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Using Python Directly
```bash
# Install dependencies
pip install -r requirements.txt

# Start daemon mode
python main.py --daemon

# Or start with debug window
python main.py --debug
```

## Configuration

The system is configured via `config.json`. Here's a complete breakdown of all configuration options:

### Camera Configuration
```json
{
  "camera": {
    "ip": "192.168.0.251",              // Camera IP address
    "port": 80,                         // HTTP/ONVIF port (usually 80)
    "username": "admin",                // Camera username
    "password": "admin",                // Camera password
    "stream_url": "rtsp://192.168.0.251:554/stream1",  // RTSP stream URL
    "onvif_port": 80,                   // ONVIF service port
    "soap_timeout": 5,                  // SOAP request timeout (seconds)
    "connection_timeout": 10,           // Connection timeout (seconds)
    "use_digest_auth": true,            // Use digest authentication
    "rtsp_transport": "tcp"             // RTSP transport protocol (tcp/udp)
  }
}
```

### Tracking Configuration
```json
{
  "tracking": {
    "confidence_threshold": 0.3,        // Minimum detection confidence (0.0-1.0)
    "dead_zone_width": 0.4,             // Horizontal dead zone (0.0-1.0)
    "dead_zone_height": 0.45,           // Vertical dead zone (0.0-1.0)
    "smoothing_factor": 0.2,            // Movement smoothing (0.0-1.0)
    "lost_tracking_timeout": 2.0,       // Time before lost tracking (seconds)
    "min_detection_size": 0.02,         // Minimum person size (0.0-1.0)
    "max_people": 3,                    // Maximum people to track
    "lock_distance_threshold": 0.15,    // Lock distance threshold
    "track_buffer_frames": 60,          // Tracking buffer size
    "min_track_hits": 1,                // Minimum hits to confirm track
    "iou_threshold": 0.2                // Intersection over Union threshold
  }
}
```

### PTZ Control Configuration
```json
{
  "ptz": {
    "pan_speed": 0.15,                  // Pan movement speed (0.0-1.0)
    "tilt_speed": 0.3,                  // Tilt movement speed (0.0-1.0)
    "zoom_speed": 0.2,                  // Zoom movement speed (0.0-1.0)
    "max_pan_angle": 170,               // Maximum pan angle (degrees)
    "min_pan_angle": -170,              // Minimum pan angle (degrees)
    "max_tilt_angle": 20,               // Maximum tilt angle (degrees)
    "min_tilt_angle": -90,              // Minimum tilt angle (degrees)
    "invert_pan": false,                // Invert pan direction
    "movement_threshold": 0.1,          // Minimum movement threshold
    "continuous_move": true             // Use continuous movement
  }
}
```

### OSC Configuration
```json
{
  "osc": {
    "enabled": true,                    // Enable OSC server
    "host": "0.0.0.0",                  // OSC server bind address
    "port": 8081,                       // OSC server port
    "client_host": "127.0.0.1",         // OSC client address (for status updates)
    "client_port": 8082                 // OSC client port
  }
}
```

### System Configuration
```json
{
  "system": {
    "debug": false,                     // Enable debug logging
    "log_level": "INFO",                // Log level (DEBUG, INFO, WARNING, ERROR)
    "frame_skip": 1,                    // Process every N frames (performance)
    "gpu_acceleration": true,           // Use GPU for AI inference
    "model_path": "yolov8n.pt"          // Path to YOLO model file
  }
}
```

## Usage

### Command Line Options
```bash
# Start in daemon mode (background with API/OSC servers)
python main.py --daemon

# Start with debug window
python main.py --debug

# Start without GUI window
python main.py --no-window

# Move camera to home position and exit
python main.py --home

# Move camera to specific preset and exit
python main.py --preset 2

# Use custom config file
python main.py --config my_config.json
```

### Docker Usage
```bash
# Start in background
docker compose up -d

# Start with development profile (debug mode)
docker compose --profile dev up

# View real-time logs
docker compose logs -f

# Restart after config changes
docker compose restart

# Stop and remove container
docker compose down
```

## OSC Control Reference

The system provides OSC (Open Sound Control) interface for real-time control, perfect for integration with control systems like Bitfocus Companion.

### OSC Commands (Port 8081)

The system accepts OSC messages for real-time control. All commands use UDP.

#### Control Commands

| OSC Address | Arguments | Description |
|-------------|-----------|-------------|
| `/ptz/start` | None | Start person tracking |
| `/ptz/stop` | None | Stop person tracking |
| `/ptz/relock` | None | Re-lock onto primary person |
| `/ptz/preset` | `int` preset_number | Go to specific preset |
| `/ptz/pan` | `float` angle | Pan to specific angle |
| `/ptz/home` | None | Return to home position |

#### Query Commands

| OSC Address | Arguments | Description |
|-------------|-----------|-------------|
| `/ptz/status` | None | Request status update |
| `/ptz/people_count` | None | Request people count |
| `/ptz/lock_status` | None | Request lock status |

#### Status Updates (Outbound)

The system sends status updates to the configured OSC client:

| OSC Address | Arguments | Description |
|-------------|-----------|-------------|
| `/ptz/status` | `string` status | Overall system status |
| `/ptz/people_count` | `int` count | Number of people detected |
| `/ptz/tracking_status` | `bool` tracking | Tracking active status |
| `/ptz/lock_status` | `bool` locked, `int` person_id | Lock status and target |

### Example OSC Usage

**Using Python with python-osc:**
```python
from pythonosc import udp_client

# Create OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 8081)

# Start tracking
client.send_message("/ptz/start", None)

# Lock onto person
client.send_message("/ptz/relock", None)

# Pan to 30 degrees
client.send_message("/ptz/pan", 30.0)

# Go to preset 2
client.send_message("/ptz/preset", 2)

# Stop tracking
client.send_message("/ptz/stop", None)
```

## Bitfocus Companion Integration

The PTZ Camera Tracking System integrates with Bitfocus Companion using OSC protocol for professional streaming and live event control.

### OSC Module Setup

1. **Add OSC Module** in Companion
2. **Configure OSC Settings:**
   - **Label**: PTZ Camera Tracker
   - **Target IP**: `127.0.0.1` (or Docker host IP)
   - **Target Port**: `8081`
   - **Protocol**: `UDP`
   - **Listen Port**: `8082`

### Button Configuration Examples

#### Start/Stop Tracking Button
**Button 1: Start Tracking**
- **Action**: OSC Send Message
- **OSC Path**: `/ptz/start`
- **Button Text**: `START\nTRACKING`
- **Button Color**: Green

**Button 2: Stop Tracking**
- **Action**: OSC Send Message
- **OSC Path**: `/ptz/stop`
- **Button Text**: `STOP\nTRACKING`
- **Button Color**: Red

#### Person Lock Button
**Button 3: Lock Person**
- **Action**: OSC Send Message
- **OSC Path**: `/ptz/relock`
- **Button Text**: `LOCK\nPERSON`
- **Button Color**: Yellow

#### Preset Buttons
**Button 4: Home Position**
- **Action**: OSC Send Message
- **OSC Path**: `/ptz/preset`
- **Value**: `1`
- **Button Text**: `HOME\nPOSITION`

**Button 5: Stage Left**
- **Action**: OSC Send Message
- **OSC Path**: `/ptz/pan`
- **Value**: `-45.0`
- **Button Text**: `STAGE\nLEFT`

**Button 6: Stage Right**
- **Action**: OSC Send Message
- **OSC Path**: `/ptz/pan`
- **Value**: `45.0`
- **Button Text**: `STAGE\nRIGHT`

#### Status Query Button
**Button 7: Status Check**
- **Action**: OSC Send Message
- **OSC Path**: `/ptz/status`
- **Button Text**: `STATUS\nCHECK`

#### Status Monitoring with Feedbacks
Use OSC status responses to monitor system state:

**Feedback Configuration:**
- **Trigger**: Receive OSC message on `/ptz/tracking_status`
- **Parse Value**: Check boolean tracking state
- **Button Color**: Green if tracking, Red if stopped

### Advanced Companion Features

#### Using Variables for Dynamic Control
Create custom variables to store tracking state from OSC responses:

```javascript
// In Companion Custom Variables  
$(internal:custom_ptz_tracking)     // true/false from OSC feedback
$(internal:custom_ptz_people_count) // number from OSC feedback
$(internal:custom_ptz_locked)       // true/false from OSC feedback
```

#### Automated Sequences
Create button sequences for complex operations:

**Button: Auto-Track Sequence**
1. Send `/ptz/stop` - Stop current tracking
2. Wait 500ms
3. Send `/ptz/home` - Go to home position  
4. Wait 1000ms
5. Send `/ptz/start` - Start tracking
6. Wait 2000ms
7. Send `/ptz/relock` - Lock onto person

#### Status Monitoring with Feedbacks
Use OSC status responses to monitor system state:

**Feedback Configuration:**
- **Trigger**: Receive OSC message on `/ptz/tracking_status`
- **Parse Value**: Check boolean tracking state
- **Button Color**: Green if tracking, Red if stopped

### Error Handling

Configure error handling in Companion:

**OSC Timeout**: Set to 5 seconds for OSC message responses
**Retry Logic**: Retry failed OSC messages 2 times  
**Error Feedback**: Show red button color on OSC timeout

### Production Deployment Notes

1. **Network Configuration**: Ensure Companion can reach the PTZ system
2. **Firewall Rules**: Open port 8081 (OSC) for UDP traffic
3. **Docker Networking**: Use `host` network mode or expose port 8081
4. **Backup Controls**: Always have manual camera controls as backup
5. **Health Monitoring**: Use `/ptz/status` OSC command for system monitoring

## Technical Notes

### Architecture Overview
- **AI Detection**: YOLOv8 neural network with GPU acceleration
- **Tracking Algorithm**: Multi-object tracking with Kalman filtering
- **Camera Control**: ONVIF protocol with VISCA commands
- **Movement Smoothing**: Exponential smoothing with configurable dead zones
- **Container Support**: Docker with NVIDIA GPU runtime support

### Performance Optimization
- **Frame Processing**: Configurable frame skipping for performance
- **GPU Acceleration**: CUDA-enabled PyTorch and OpenCV
- **Memory Management**: Efficient buffer management for real-time processing
- **Network Optimization**: Keep-alive connections and request pooling

### Error Handling & Recovery
- **Automatic Fallback**: Returns to preset position on tracking loss
- **Connection Recovery**: Automatic reconnection to camera and services
- **Graceful Degradation**: System continues operation with component failures
- **Health Monitoring**: Built-in health checks and status reporting

### Security Considerations
- **Camera Credentials**: Secure storage of camera authentication
- **API Access**: CORS-enabled for web integration
- **Network Security**: Configurable bind addresses for network isolation
- **Container Security**: Non-root user execution in Docker

## Troubleshooting

### Camera Connection Issues

**Problem**: Cannot connect to camera
```
ERROR: Failed to initialize PTZ controller
```

**Solutions**:
1. Verify camera IP and credentials in `config.json`
2. Test camera web interface: `http://[camera_ip]`
3. Check ONVIF service: `http://[camera_ip]/onvif/device_service`
4. Verify network connectivity: `ping [camera_ip]`
5. Check firewall settings on both camera and host

**Problem**: RTSP stream connection fails
```
INFO: [FAILED] Could not connect to RTSP stream
```

**Solutions**:
1. Test stream URL manually: `ffplay rtsp://camera_ip:554/stream1`
2. Try alternative stream paths: `/stream0`, `/live`, `/h264`
3. Check RTSP port (usually 554)
4. Verify camera streaming settings
5. Try different transport protocols (TCP/UDP)

### Person Detection Issues

**Problem**: No people detected
```
INFO: Detection stats: 0 people detected
```

**Solutions**:
1. Check camera view and lighting conditions
2. Lower `confidence_threshold` in config (try 0.2)
3. Verify `min_detection_size` is appropriate
4. Test with debug window: `python main.py --debug`
5. Check GPU/CUDA availability for AI acceleration

**Problem**: False detections or erratic tracking
```
WARNING: Multiple people detected, selecting primary
```

**Solutions**:
1. Increase `confidence_threshold` to reduce false positives
2. Adjust `iou_threshold` for better tracking consistency
3. Increase `smoothing_factor` for smoother movement
4. Configure appropriate `dead_zone_width`

### Movement and Control Issues

**Problem**: Camera moves too quickly or erratically
```
INFO: Camera pan command executed
```

**Solutions**:
1. Reduce `pan_speed` and `tilt_speed` values
2. Increase `smoothing_factor` (0.3-0.5)
3. Increase `dead_zone_width` to reduce sensitivity
4. Enable `continuous_move` for smoother operation

**Problem**: Camera doesn't respond to commands
```
ERROR: PTZ command failed
```

**Solutions**:
1. Check VISCA protocol support on camera
2. Verify ONVIF credentials and permissions
3. Test manual camera control via web interface
4. Check `movement_threshold` setting
5. Review camera angle limits in config

### OSC and Integration Issues

**Problem**: OSC commands not responding
```
ERROR: OSC server error
```

**Solutions**:
1. Check if daemon is running: `docker compose ps`
2. Verify port accessibility: Test with OSC client on port 8081
3. Check firewall rules for port 8081 UDP traffic
4. Review OSC logs: `docker compose logs -f`
5. Test with simple OSC send commands first

**Problem**: Bitfocus Companion integration issues

**Solutions**:
1. Verify Companion can reach OSC endpoint on port 8081
2. Check OSC module configuration in Companion
3. Use `/ptz/status` command to test connectivity
4. Monitor OSC message responses and error messages
5. Ensure UDP port 8081 is not blocked by firewall

### Docker-Specific Issues

**Problem**: GPU acceleration not working
```
WARNING: CUDA not available, using CPU
```

**Solutions**:
1. Install NVIDIA Docker runtime
2. Verify GPU support: `docker run --gpus all nvidia/cuda:11.0-base nvidia-smi`
3. Update docker-compose.yml GPU configuration
4. Check NVIDIA driver version compatibility

**Problem**: Container keeps restarting
```
FATAL: exception not rethrown
```

**Solutions**:
1. Check container logs: `docker compose logs`
2. Verify all required files are present
3. Check config.json syntax and values
4. Ensure camera is accessible from container
5. Review resource limits in docker-compose.yml

### Performance Issues

**Problem**: High CPU/Memory usage

**Solutions**:
1. Increase `frame_skip` to process fewer frames
2. Enable GPU acceleration for AI processing
3. Reduce `max_people` if not needed
4. Lower camera resolution or frame rate
5. Optimize `track_buffer_frames` size

**Problem**: Delayed or laggy tracking

**Solutions**:
1. Reduce video stream latency settings
2. Use TCP transport for RTSP if UDP is unreliable
3. Minimize network hops between camera and system
4. Optimize `movement_cooldown` timing
5. Consider dedicated network for camera traffic

## Development and Debugging

### Debug Mode
```bash
# Run with debug window to see real-time detection
python main.py --debug

# Docker with debug logs
docker compose logs -f --tail=100
```

### Log Levels
Configure logging in `config.json`:
```json
{
  "system": {
    "debug": true,
    "log_level": "DEBUG"
  }
}
```

### Testing Tools
```bash
# Test camera connection only
python -m src.camera_controller

# Test person detection with webcam
python -m src.person_tracker

# Full system test
python test_system.py
```

## License

MIT License - see LICENSE file for details.

## Contributing
You are welcome to contribute. Just please do so in a new branch and submit a merge request

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include logs and configuration details

## Changelog

### v2.0.0 (Current)
- Added Docker containerization with GPU support
- Implemented YOLOv8 AI detection with CUDA acceleration  
- Enhanced API with comprehensive REST endpoints
- Added OSC protocol support for real-time control
- Improved multi-person tracking with ID persistence
- Added Bitfocus Companion integration examples
- Enhanced configuration system with full documentation
- Added comprehensive error handling and recovery
- Implemented health monitoring and status reporting

### v1.0.0
- Initial release with OpenCV HOG detection
- Basic ONVIF PTZ camera control
- Simple tracking algorithm
- Basic API endpoints