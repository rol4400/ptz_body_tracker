#!/usr/bin/env python3
"""
PTZ Camera Controller - VISCA Protocol
Handles communication with PTZ cameras via Sony VISCA protocol
"""

import socket
import time
import logging
import asyncio
from typing import Tuple, Optional
import threading

class PTZController:
    """PTZ Camera Controller using VISCA protocol"""
    
    def __init__(self, config: dict):
        """
        Initialize camera controller
        
        Args:
            config: Camera configuration dictionary
        """
        self.config = config
        self.camera_config = config['camera']
        self.ptz_config = config['ptz']
        self.logger = logging.getLogger(__name__)
        
        # VISCA connection settings
        self.ip = self.camera_config['ip']
        self.port = 5678  # VISCA port discovered for this camera
        
        self.socket = None
        self.connected = False
        
        # VISCA settings
        self.default_pan_speed = 0x10   # Medium speed
        self.default_tilt_speed = 0x10  # Medium speed
        self.max_pan_speed = 0x18       # Maximum pan speed
        self.max_tilt_speed = 0x14      # Maximum tilt speed
        
        # Position tracking
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.current_zoom = 0.0
        self.position_lock = threading.Lock()
        
        # Movement smoothing
        self.target_pan = 0.0
        self.movement_lock = asyncio.Lock()
        
    async def initialize(self) -> bool:
        """Initialize camera connection"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.ip, self.port))
            
            # Test connection with a simple command that we know works
            # Try pan right then stop (non-destructive test)
            test_commands = [
                # [0x81, 0x01, 0x06, 0x01, 0x10, 0x10, 0x02, 0x03, 0xFF],  # Pan right
                [0x81, 0x01, 0x06, 0x01, 0x10, 0x10, 0x03, 0x03, 0xFF]   # Stop
            ]
            
            connection_ok = False
            for cmd in test_commands:
                try:
                    cmd_bytes = bytes(cmd)
                    self.socket.send(cmd_bytes)
                    self.socket.settimeout(2.0)
                    response = self.socket.recv(1024)
                    if response:  # Any response indicates the camera is listening
                        connection_ok = True
                        break
                except:
                    continue
            
            if connection_ok:
                self.connected = True
                self.logger.info(f"[OK] Connected to VISCA camera at {self.ip}:{self.port}")
                
                # Don't get initial position or go to home - just connect
                await self.get_current_position()
                return True
            else:
                self.logger.error("Camera not responding to VISCA commands")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to camera: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    def disconnect(self):
        """Disconnect from camera"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            self.connected = False
            self.logger.info("Disconnected from camera")
    
    def _send_command(self, command: list) -> bool:
        """Send VISCA command and check for ACK"""
        if not self.socket:
            return False
            
        try:
            cmd_bytes = bytes(command)
            self.socket.send(cmd_bytes)
            
            # Wait for response
            self.socket.settimeout(2.0)
            response = self.socket.recv(1024)
            
            # Check for ACK (90 41 FF) or completion (90 5x FF)
            if response and len(response) >= 3:
                if (response[0] == 0x90 and response[1] == 0x41) or \
                   (response[0] == 0x90 and response[1] == 0x51) or \
                   (response[0] == 0x90 and response[1] >= 0x50 and response[1] <= 0x5F):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending VISCA command: {e}")
            return False
    
    def _query_command(self, command: list) -> Optional[bytes]:
        """Send VISCA query command and get response data"""
        if not self.socket:
            return None
            
        try:
            cmd_bytes = bytes(command)
            self.socket.send(cmd_bytes)
            
            # Wait for response
            self.socket.settimeout(2.0)
            response = self.socket.recv(1024)
            
            # Check for valid response (90 50 ...)
            if response and len(response) >= 4 and response[0] == 0x90 and response[1] == 0x50:
                return response
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error sending VISCA query: {e}")
            return None
    
    def _visca_units_to_degrees(self, visca_units: int) -> float:
        """Convert VISCA units to degrees (approximate)"""
        # This conversion factor may need calibration for your specific camera
        return visca_units * 0.01
    
    def _degrees_to_visca_units(self, degrees: float) -> int:
        """Convert degrees to VISCA units (approximate)"""
        return int(degrees / 0.01)
    
    async def get_current_position(self) -> Tuple[float, float, float]:
        """Get current PTZ position"""
        if not self.connected:
            return (self.current_pan, self.current_tilt, self.current_zoom)
        
        # Pan-tilt position inquiry: 81 09 06 12 FF
        response = self._query_command([0x81, 0x09, 0x06, 0x12, 0xFF])
        
        if response and len(response) >= 11:
            # Response: 90 50 0w 0w 0w 0w 0z 0z 0z 0z FF
            # Extract pan position (4 nibbles)
            pan_units = (response[2] << 12) | (response[3] << 8) | (response[4] << 4) | response[5]
            # Extract tilt position (4 nibbles)
            tilt_units = (response[6] << 12) | (response[7] << 8) | (response[8] << 4) | response[9]
            
            # Convert from unsigned to signed
            if pan_units > 0x8000:
                pan_units -= 0x10000
            if tilt_units > 0x8000:
                tilt_units -= 0x10000
                
            with self.position_lock:
                self.current_pan = self._visca_units_to_degrees(pan_units)
                self.current_tilt = self._visca_units_to_degrees(tilt_units)
        
        return (self.current_pan, self.current_tilt, self.current_zoom)
    
    async def move_to_position(self, pan: float, tilt: Optional[float] = None, zoom: Optional[float] = None, speed: Optional[float] = None, pan_only: bool = False):
        """Move to absolute position"""
        async with self.movement_lock:
            if not self.connected:
                return
            
            # If pan_only is True, keep current tilt position and don't send tilt commands
            if pan_only:
                tilt = self.current_tilt
            elif tilt is None:
                tilt = self.current_tilt
                
            if zoom is None:
                zoom = self.current_zoom
            if speed is None:
                speed = 0.5  # Default medium speed
            
            # Clamp pan to limits
            pan = max(self.ptz_config['min_pan_angle'], 
                     min(self.ptz_config['max_pan_angle'], pan))
            
            # For pan-only movement, use pan command instead of absolute position
            if pan_only:
                # Use pan command that doesn't affect tilt
                await self.pan_to(pan, speed=speed)
                return
            
            # Convert degrees to VISCA units
            pan_units = self._degrees_to_visca_units(pan)
            tilt_units = self._degrees_to_visca_units(tilt)
            
            # Clamp to valid VISCA ranges
            pan_units = max(-0x8000, min(0x7FFF, pan_units))
            tilt_units = max(-0x8000, min(0x7FFF, tilt_units))
            
            # Convert to unsigned for transmission
            if pan_units < 0:
                pan_units += 0x10000
            if tilt_units < 0:
                tilt_units += 0x10000
            
            # Convert speed (0.0-1.0) to VISCA speed (1-24 for pan, 1-20 for tilt)
            visca_pan_speed = max(1, min(int(speed * self.max_pan_speed), self.max_pan_speed))
            visca_tilt_speed = max(1, min(int(speed * self.max_tilt_speed), self.max_tilt_speed))
            
            # Extract nibbles for pan position
            pan_nibbles = [
                (pan_units >> 12) & 0x0F,
                (pan_units >> 8) & 0x0F,
                (pan_units >> 4) & 0x0F,
                pan_units & 0x0F
            ]
            
            # Extract nibbles for tilt position
            tilt_nibbles = [
                (tilt_units >> 12) & 0x0F,
                (tilt_units >> 8) & 0x0F,
                (tilt_units >> 4) & 0x0F,
                tilt_units & 0x0F
            ]
            
            # Absolute position command: 81 01 06 02 VV WW 0Y 0Y 0Y 0Y 0Z 0Z 0Z 0Z FF
            command = [0x81, 0x01, 0x06, 0x02, 
                      visca_pan_speed, visca_tilt_speed] + \
                      pan_nibbles + tilt_nibbles + [0xFF]
            
            if self._send_command(command):
                self.current_pan = pan
                self.current_tilt = tilt
                self.logger.debug(f"Moved to position: Pan={pan:.1f}°, Tilt={tilt:.1f}°")
    
    async def pan_to(self, pan_angle: float, speed: Optional[float] = None):
        """Pan to specific angle (keeping current tilt/zoom) using pan-only command"""
        async with self.movement_lock:
            if not self.connected:
                return
                
            if speed is None:
                speed = 0.5  # Default medium speed
                
            # Clamp pan to limits
            pan_angle = max(self.ptz_config['min_pan_angle'], 
                           min(self.ptz_config['max_pan_angle'], pan_angle))
            
            # Convert degrees to VISCA units
            pan_units = self._degrees_to_visca_units(pan_angle)
            
            # Clamp to valid VISCA ranges
            pan_units = max(-0x8000, min(0x7FFF, pan_units))
            
            # Convert to unsigned for transmission
            if pan_units < 0:
                pan_units += 0x10000
            
            # Convert speed (0.0-1.0) to VISCA speed (1-24 for pan)
            visca_pan_speed = max(1, min(int(speed * self.max_pan_speed), self.max_pan_speed))
            
            # Extract nibbles for pan position
            pan_nibbles = [
                (pan_units >> 12) & 0x0F,
                (pan_units >> 8) & 0x0F,
                (pan_units >> 4) & 0x0F,
                pan_units & 0x0F
            ]
            
            # Pan absolute position command (without affecting tilt): 81 01 06 03 VV 0Y 0Y 0Y 0Y FF
            command = [0x81, 0x01, 0x06, 0x03, visca_pan_speed] + pan_nibbles + [0xFF]
            
            if self._send_command(command):
                self.current_pan = pan_angle
                self.logger.debug(f"Pan to: {pan_angle:.1f}° (tilt unchanged)")
    
    async def pan_left(self, speed: float = 0.3):
        """Pan camera left (continuous movement)"""
        if not self.connected:
            return
            
        # Convert speed (0.0-1.0) to VISCA speed (1-24)
        visca_pan_speed = max(1, min(int(speed * self.max_pan_speed), self.max_pan_speed))
        
        # Pan left command: 81 01 06 01 VV 00 01 03 FF
        command = [0x81, 0x01, 0x06, 0x01, visca_pan_speed, 0x00, 0x01, 0x03, 0xFF]
        self._send_command(command)
        self.logger.debug(f"Pan left at speed {speed:.2f}")
    
    async def pan_right(self, speed: float = 0.3):
        """Pan camera right (continuous movement)"""
        if not self.connected:
            return
            
        # Convert speed (0.0-1.0) to VISCA speed (1-24)
        visca_pan_speed = max(1, min(int(speed * self.max_pan_speed), self.max_pan_speed))
        
        # Pan right command: 81 01 06 01 VV 00 02 03 FF
        command = [0x81, 0x01, 0x06, 0x01, visca_pan_speed, 0x00, 0x02, 0x03, 0xFF]
        self._send_command(command)
        self.logger.debug(f"Pan right at speed {speed:.2f}")
    
    async def goto_preset(self, preset_number: Optional[int] = None):
        """Go to preset position"""
        if not self.connected:
            return
        
        if preset_number is None:
            preset_number = self.ptz_config.get('default_preset', 1)
        
        assert preset_number is not None  # For type checker
        if preset_number < 0 or preset_number > 127:
            self.logger.error(f"Invalid preset number: {preset_number}")
            return
        
        # Recall preset: 81 01 04 3F 02 pp FF
        command = [0x81, 0x01, 0x04, 0x3F, 0x02, preset_number, 0xFF]
        
        if self._send_command(command):
            self.logger.info(f"Recalled preset {preset_number}")
            # Wait a moment then update position
            await asyncio.sleep(1.0)
            await self.get_current_position()
    
    async def goto_home(self):
        """Go to home position"""
        if not self.connected:
            return
        
        # Home: 81 01 06 04 FF
        command = [0x81, 0x01, 0x06, 0x04, 0xFF]
        
        if self._send_command(command):
            self.logger.info("Moving to home position")
            # Wait for movement to complete then update position
            await asyncio.sleep(3.0)
            await self.get_current_position()
    
    async def stop_movement(self):
        """Stop current PTZ movement"""
        if not self.connected:
            return
        
        # Stop: 81 01 06 01 VV WW 03 03 FF
        command = [0x81, 0x01, 0x06, 0x01, 
                  self.default_pan_speed, self.default_tilt_speed, 
                  0x03, 0x03, 0xFF]
        
        if self._send_command(command):
            self.logger.debug("Stopped PTZ movement")
    
    def calculate_pan_for_position(self, x_normalized: float) -> float:
        """Calculate pan angle for normalized X position (0.0 to 1.0)"""
        # Convert normalized position to pan angle
        # x_normalized: 0.0 = left edge, 0.5 = center, 1.0 = right edge
        
        pan_range = self.ptz_config['max_pan_angle'] - self.ptz_config['min_pan_angle']
        
        # Check if pan should be inverted (for cameras facing different directions)
        invert_pan = self.ptz_config.get('invert_pan', False)
        
        if invert_pan:
            # INVERTED: When person is on left (x=0), camera should pan right (positive)
            #          When person is on right (x=1), camera should pan left (negative)
            target_pan = self.ptz_config['max_pan_angle'] - (x_normalized * pan_range)
        else:
            # NORMAL: When person is on left (x=0), camera should pan left (negative)
            #        When person is on right (x=1), camera should pan right (positive)
            target_pan = self.ptz_config['min_pan_angle'] + (x_normalized * pan_range)
        
        return target_pan
    
    def calculate_tilt_for_position(self, y_normalized: float) -> float:
        """Calculate tilt angle for normalized Y position (0.0 to 1.0)"""
        # Convert normalized position to tilt angle
        # y_normalized: 0.0 = top edge, 0.5 = center, 1.0 = bottom edge
        # Use a much smaller tilt range for smooth tracking (±15° from center)
        
        # Define a reasonable tilt range for person tracking (not the full camera range)
        tracking_tilt_range = 30.0  # ±15° from center position
        center_tilt = 0.0  # Assume level is center for person tracking
        
        # Convert y position to tilt offset from center
        # When person is at top (y=0), tilt up slightly (+15°)
        # When person is at center (y=0.5), no tilt (0°)
        # When person is at bottom (y=1), tilt down slightly (-15°)
        tilt_offset = (0.5 - y_normalized) * tracking_tilt_range
        target_tilt = center_tilt + tilt_offset
        
        # Clamp to reasonable limits for person tracking
        min_tilt = self.ptz_config.get('min_tilt_angle', -90.0)
        max_tilt = self.ptz_config.get('max_tilt_angle', 20.0)
        target_tilt = max(min_tilt, min(max_tilt, target_tilt))
        
        return target_tilt
    
    def is_in_dead_zone(self, x_normalized: float) -> bool:
        """Check if position is in the center dead zone"""
        dead_zone_width = self.config['tracking']['dead_zone_width']
        center_x = 0.5
        dead_zone_left = center_x - (dead_zone_width / 2)
        dead_zone_right = center_x + (dead_zone_width / 2)
        
        return dead_zone_left <= x_normalized <= dead_zone_right
    
    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect()


# Test function
async def test_visca_controller():
    """Test the VISCA PTZ controller"""
    # Mock config for testing
    config = {
        'camera': {
            'ip': '192.168.0.251',
            'port': 80,
            'username': 'admin',
            'password': 'admin'
        },
        'ptz': {
            'min_pan_angle': -90.0,
            'max_pan_angle': 90.0,
            'pan_speed': 0.5,
            'default_preset': 1,
            'home_position': {'pan': 0.0, 'tilt': 0.0, 'zoom': 0.0}
        },
        'tracking': {
            'dead_zone_width': 0.2
        }
    }
    
    logging.basicConfig(level=logging.INFO)
    
    controller = PTZController(config)
    
    print("Testing VISCA PTZ controller...")
    if await controller.initialize():
        print("✓ Connected successfully")
        
        # Test position query
        pan, tilt, zoom = await controller.get_current_position()
        print(f"Current position: Pan={pan:.2f}°, Tilt={tilt:.2f}°, Zoom={zoom:.2f}")
        
        # Test movements
        print("Testing movements...")
        
        print("Pan to 10°...")
        await controller.pan_to(10.0, speed=0.3)
        await asyncio.sleep(2)
        
        print("Pan to -10°...")
        await controller.pan_to(-10.0, speed=0.3)
        await asyncio.sleep(2)
        
        print("Moving to home...")
        await controller.goto_home()
        await asyncio.sleep(3)
        
        controller.disconnect()
        print("✓ Test completed successfully")
        
    else:
        print("❌ Failed to connect to camera")


if __name__ == "__main__":
    asyncio.run(test_visca_controller())