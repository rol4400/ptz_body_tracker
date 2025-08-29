#!/usr/bin/env python3
"""
PTZ Camera Tracking System - Main Application
Tracks people using PTZ camera with YOLOv8 detection and VISCA control
"""

import cv2
import asyncio
import logging
import time
import json
import sys
import argparse
from pathlib import Path
import signal
import threading
from typing import Optional
from queue import Queue, Empty

from src.person_tracker import BodyTracker
from src.camera_controller import PTZController
from src.debug_window import DebugWindow
from src.osc_controller import OSCController


class LatencyOptimizedCapture:
    """Video capture with minimal latency using threading"""
    
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.capture = None
        self.frame_queue = Queue(maxsize=2)  # Small queue
        self.latest_frame = None
        self.capture_thread = None
        self.running = False
        
    def start(self):
        """Start capture thread"""
        self.capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            return False
            
        # Optimize for low latency and full resolution
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 25)
        
        # Ensure we get the full resolution from the camera stream
        # Don't set frame width/height to let the camera provide its native resolution
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return True
    
    def _capture_loop(self):
        """Continuous capture loop that keeps only the latest frame"""
        while self.running and self.capture:
            ret, frame = self.capture.read()
            if ret:
                # Keep only the absolute latest frame to minimize latency
                self.latest_frame = frame
                
                # Clear the queue and put the latest frame
                try:
                    # Remove any stale frames
                    while not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    # Add the fresh frame
                    self.frame_queue.put_nowait(frame)
                except Exception:
                    pass
                except:
                    pass  # Queue full, skip frame
    
    def read(self):
        """Get the latest frame"""
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except Empty:
            # Return last known frame if queue is empty
            if self.latest_frame is not None:
                return True, self.latest_frame
            return False, None
    
    def stop(self):
        """Stop capture"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.capture:
            self.capture.release()
    
    def get(self, prop):
        """Get capture property"""
        if self.capture:
            return self.capture.get(prop)
        return 0


class PTZTrackingSystem:
    """Main PTZ tracking system"""
    
    def __init__(self, config: dict, show_debug: bool = False):
        self.config = config
        self.show_debug = show_debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.body_tracker = BodyTracker(config)
        self.ptz_controller = PTZController(config)
        
        # Video capture
        self.video_capture: Optional[LatencyOptimizedCapture] = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Debug window
        if self.show_debug:
            self.debug_window = DebugWindow("PTZ Camera Tracker")
            self.show_help = False
        else:
            self.debug_window = None
        
        # OSC server only
        self.osc_controller = None
        
        # Enable OSC server based on config
        osc_config = config.get('osc', {})
        
        if osc_config.get('enabled', True):
            self.osc_controller = OSCController(config, self)
        
        # Tracking state
        self.is_running = False
        self.is_tracking = False
        self.is_paused = False
        self.primary_person_id = None
        
        # Movement control
        self.last_movement_time = 0
        self.movement_cooldown = 0.1  # Reduced to 0.1 seconds for very responsive movement
        self.is_camera_moving = False  # Track if camera is currently moving
        self.dead_zone_width = config.get('tracking', {}).get('dead_zone_width', 0.2)
        
        # Person tracking stability
        self.person_selection_frames = 0
        self.min_selection_frames = 5  # Reduced from 30 to 5 frames for faster re-selection
        self.lost_person_timeout = 60  # Increased from 30 to 60 frames to keep tracking longer
        self.frame_skip_count = 0
        self.frame_skip_target = self.config.get('system', {}).get('frame_skip', 1)
        
        # Remember last position for quick re-locking
        self.last_primary_position = None
        self.last_primary_id = None
        
        # Movement direction control
        self.current_direction = None
        self.direction_change_cooldown = 0.3  # Reduced from 1.0 for faster direction changes
        
        # No-lock preset recall
        self.last_lock_time = time.time()  # When we last had a person locked
        self.no_lock_timeout = 2.0  # Seconds before going to preset
        self.preset_recalled = False  # Track if we've already recalled preset to avoid spam
        
    async def initialize(self):
        """Initialize all components"""
        self.logger.info("Initializing PTZ tracking system...")
        
        # Initialize PTZ controller
        if not await self.ptz_controller.initialize():
            raise Exception("Failed to initialize PTZ controller")
        
        # Store camera config for later video capture initialization
        self.camera_config = self.config['camera']
        
        # Start OSC server
        if self.osc_controller:
            self.osc_controller.start()
            self.logger.info("OSC controller started")
        
        self.logger.info("PTZ tracking system initialized successfully")
    
    async def initialize_video_capture(self):
        """Initialize video capture from camera RTSP stream"""
        if self.video_capture:
            return  # Already initialized
        
        camera_config = self.camera_config
        
        # Try multiple RTSP stream URLs
        rtsp_urls = [
            camera_config.get('stream_url', f"rtsp://{camera_config['ip']}:554/stream1"),
            f"rtsp://{camera_config['ip']}:554/stream0",
            f"rtsp://{camera_config['ip']}:554/live",
            f"rtsp://{camera_config['ip']}:554/h264",
            f"rtsp://{camera_config['username']}:{camera_config['password']}@{camera_config['ip']}:554/stream1"
        ]
        
        self.video_capture = None
        for rtsp_url in rtsp_urls:
            self.logger.info(f"Trying RTSP URL: {rtsp_url}")
            capture = LatencyOptimizedCapture(rtsp_url)
            
            if capture.start():
                # Test if we can actually read a frame
                time.sleep(0.5)  # Give it time to start
                ret, test_frame = capture.read()
                if ret and test_frame is not None:
                    self.logger.info(f"[OK] Successfully connected to: {rtsp_url}")
                    self.video_capture = capture
                    break
                else:
                    capture.stop()
            else:
                capture.stop()
        
        if not self.video_capture:
            self.logger.error("Failed to connect to any RTSP stream")
            raise Exception("Failed to open camera stream")
        
        # Get actual stream properties
        actual_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Stream properties: {width}x{height} @ {actual_fps:.1f} FPS")
        self.logger.info(f"Aspect ratio: {width/height:.2f} (16:9 = 1.78)")
        
        # Verify we have a proper 16:9 aspect ratio for full coverage
        aspect_ratio = width / height
        if abs(aspect_ratio - 16/9) > 0.1:
            self.logger.warning(f"Stream aspect ratio {aspect_ratio:.2f} may not be 16:9. Check camera stream settings.")
    
    def update_camera_position(self, people):
        # Update camera position based on primary person
        if not self.is_tracking or self.is_paused:
            return
        
        current_time = time.time()
        if current_time - self.last_movement_time < self.movement_cooldown:
            return
        
        # Find primary person or select one
        primary_person = None
        if self.primary_person_id:
            # Look for existing primary person
            for person in people:
                if person.id == self.primary_person_id:
                    primary_person = person
                    self.lost_person_timeout = 30  # Reset timeout when found
                    break
            
            # If primary person not found, decrement timeout
            if not primary_person:
                self.lost_person_timeout -= 1
                if self.lost_person_timeout <= 0:
                    self.logger.info(f"Lost primary person {self.primary_person_id} - resetting")
                    self.primary_person_id = None
                    self.person_selection_frames = 0
        
        if not primary_person and people:
            # First, try to find a person near the last known position
            if self.last_primary_position is not None:
                closest_person = None
                min_distance = float('inf')
                
                for person in people:
                    # Calculate distance from last known position
                    distance = ((person.center[0] - self.last_primary_position[0])**2 + 
                              (person.center[1] - self.last_primary_position[1])**2)**0.5
                    
                    if distance < min_distance and distance < 0.2:  # Within 20% of frame
                        min_distance = distance
                        closest_person = person
                
                # If we found someone close to last position, select them immediately
                if closest_person:
                    self.primary_person_id = closest_person.id
                    primary_person = closest_person
                    self.logger.info(f"Re-selected person {self.primary_person_id} near last position")
                    self.person_selection_frames = 0
            
            # If no one near last position, select largest person with reduced stability requirement
            if not primary_person:
                largest_person = max(people, key=lambda p: p.size)
                
                # Only switch to new person if we've seen them for enough frames
                if not hasattr(self, 'candidate_person_id') or self.candidate_person_id != largest_person.id:
                    self.candidate_person_id = largest_person.id
                    self.person_selection_frames = 1
                else:
                    self.person_selection_frames += 1
                    
                if self.person_selection_frames >= self.min_selection_frames:
                    self.primary_person_id = largest_person.id
                    primary_person = largest_person
                    self.logger.info(f"Selected primary person: {self.primary_person_id}")
                    self.person_selection_frames = 0
        
        if primary_person:
            # Remember the position for quick re-locking
            self.last_primary_position = primary_person.center
            self.last_primary_id = primary_person.id
            
            # Update lock time - we have a person locked
            self.last_lock_time = current_time
            self.preset_recalled = False  # Reset preset recall flag
            
            # Calculate target pan position (horizontal tracking only)
            person_x = primary_person.center[0]  # Normalized X position (0-1)
            
            # Check if person is outside horizontal dead zone (only care about X position)
            horizontal_dead_zone = not self.ptz_controller.is_in_dead_zone(person_x)
            
            # Safety check: stop if camera has been moving too long
            if self.is_camera_moving and current_time - self.last_movement_time > 1.5:  # Reduced from 3.0
                asyncio.create_task(self.ptz_controller.stop_movement())
                self.is_camera_moving = False
                self.current_direction = None
                self.logger.warning("Camera moving too long - stopping for safety")
                return
            
            if horizontal_dead_zone:
                # Use continuous movement based on which side of center person is on
                center_x = 0.5
                offset_from_center = person_x - center_x
                
                # Only move if person is significantly off-center
                if abs(offset_from_center) > 0.05:  # 5% of frame width - increased threshold
                    # Determine movement direction
                    if offset_from_center > 0:
                        # Person is on right side, need to pan right
                        desired_direction = "right"
                    else:
                        # Person is on left side, need to pan left
                        desired_direction = "left"
                    
                    # Check if we need to start movement or change direction
                    current_direction_attr = getattr(self, 'current_direction', None)
                    direction_change_time = getattr(self, 'last_direction_change', 0)
                    
                    if not self.is_camera_moving:
                        # Camera not moving, start movement
                        movement_speed = 0.03  # Even slower continuous movement
                        
                        if desired_direction == "right":
                            asyncio.create_task(self.ptz_controller.pan_right(movement_speed))
                        else:
                            asyncio.create_task(self.ptz_controller.pan_left(movement_speed))
                        
                        self.is_camera_moving = True
                        self.current_direction = desired_direction
                        self.last_movement_time = current_time
                        self.last_direction_change = current_time
                    
                    elif current_direction_attr != desired_direction and (current_time - direction_change_time) > self.direction_change_cooldown:
                        # Need to change direction, but only if enough time has passed
                        asyncio.create_task(self.ptz_controller.stop_movement())
                        
                        movement_speed = 0.03
                        if desired_direction == "right":
                            asyncio.create_task(self.ptz_controller.pan_right(movement_speed))
                        else:
                            asyncio.create_task(self.ptz_controller.pan_left(movement_speed))
                        
                        self.current_direction = desired_direction
                        self.last_direction_change = current_time
                    
                    # Update last movement time to prevent timeout
                    self.last_movement_time = current_time
                else:
                    # Person is close to center - stop movement
                    if self.is_camera_moving:
                        asyncio.create_task(self.ptz_controller.stop_movement())
                        self.is_camera_moving = False
                        self.current_direction = None
                        self.logger.info(f"Stopping - person close to center at x={person_x:.3f}")
            else:
                # Person is in dead zone - ALWAYS stop movement
                if self.is_camera_moving:
                    asyncio.create_task(self.ptz_controller.stop_movement())
                    self.is_camera_moving = False
                    self.current_direction = None
                    self.logger.info(f"Person in dead zone - stopping camera movement")
        else:
            # No primary person - check if we should go to preset
            if current_time - self.last_lock_time > self.no_lock_timeout and not self.preset_recalled:
                self.logger.info(f"No person locked for {self.no_lock_timeout} seconds - recalling preset 4")
                asyncio.create_task(self.goto_preset(4))
                self.preset_recalled = True
    
    def is_in_vertical_dead_zone(self, y_normalized: float) -> bool:
        """Check if Y position is in the vertical center dead zone"""
        dead_zone_height = self.config['tracking']['dead_zone_height']
        center_y = 0.5
        dead_zone_top = center_y - (dead_zone_height / 2)
        dead_zone_bottom = center_y + (dead_zone_height / 2)
        
        return dead_zone_top <= y_normalized <= dead_zone_bottom
    
    async def run(self):
        """Main tracking loop"""
        self.is_running = True
        self.logger.info("Starting PTZ tracking...")
        
        try:
            while self.is_running:
                # If tracking is not active, just sleep to save resources
                if not self.is_tracking:
                    await asyncio.sleep(0.1)  # Minimal resource usage when not tracking
                    continue
                
                # Capture frame only when tracking is active
                if self.video_capture is None:
                    self.logger.error("Video capture is None")
                    break
                    
                ret, frame = self.video_capture.read()
                if not ret or frame is None:
                    self.logger.warning("Failed to read frame from camera")
                    await asyncio.sleep(0.1)
                    continue
                
                # Skip frames for performance if configured
                self.frame_skip_count += 1
                if self.frame_skip_count < self.frame_skip_target:
                    continue
                self.frame_skip_count = 0
                
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                if not self.is_paused:
                    # Detect people
                    people = self.body_tracker.detect_people(frame)
                    
                    # Log detection info every 5 seconds (150 frames at 30fps)
                    if hasattr(self, 'log_frame_count'):
                        self.log_frame_count += 1
                    else:
                        self.log_frame_count = 1
                    
                    if self.log_frame_count % 150 == 0:  # Every 5 seconds
                        if len(people) > 0:
                            self.logger.info(f"Status: Tracking {len(people)} people - Primary: {[f'ID:{p.id} conf:{p.confidence:.2f} pos:({p.center[0]:.2f},{p.center[1]:.2f})' for p in people]}")
                        else:
                            self.logger.info("Status: No people detected")
                    
                    # Update camera position
                    self.update_camera_position(people)
                else:
                    people = []
                
                # Show debug window if enabled
                if self.debug_window:
                    key = self.debug_window.show_frame(
                        frame, 
                        people, 
                        self.primary_person_id,
                        self.show_help
                    )
                    
                    action = self.debug_window.handle_key(key)
                    if action == 'quit':
                        break
                    elif action == 'lock':
                        await self.lock_primary_person()
                    elif action == 'toggle_help':
                        self.show_help = not self.show_help
                    elif action == 'pause':
                        self.is_paused = not self.is_paused
                        status = "paused" if self.is_paused else "resumed"
                        self.logger.info(f"Tracking {status}")
                
                await asyncio.sleep(0.04)  # Target ~25 FPS to match fps_limit
                
        except Exception as e:
            self.logger.error(f"Error in tracking loop: {e}")
        finally:
            await self.cleanup()
    
    async def start_tracking(self):
        """Start person tracking"""
        # Initialize video capture if not already done
        if not self.video_capture:
            self.logger.info("Initializing video capture for tracking...")
            await self.initialize_video_capture()
        
        if not self.is_running:
            # Start the main processing loop if not already running
            self.is_running = True
            asyncio.create_task(self.run())
        
        self.is_tracking = True
        self.last_lock_time = time.time()  # Reset no-lock timer when starting tracking
        self.preset_recalled = False  # Reset preset recall flag
        self.logger.info("Person tracking started")
    
    async def stop_tracking(self):
        """Stop person tracking"""
        self.is_tracking = False
        self.primary_person_id = None
        self.last_lock_time = time.time()  # Reset no-lock timer when stopping
        self.preset_recalled = False  # Reset preset recall flag
        await self.ptz_controller.stop_movement()
        self.logger.info("Person tracking stopped")
    
    async def lock_primary_person(self):
        """Lock onto the current primary person"""
        if self.primary_person_id:
            self.logger.info(f"Locked onto primary person: {self.primary_person_id}")
        else:
            self.logger.warning("No primary person to lock onto")
    
    async def goto_home(self):
        """Move camera to home position"""
        await self.ptz_controller.goto_home()
        self.logger.info("Camera moved to home position")
    
    async def goto_preset(self, preset_number: int = 1):
        """Move camera to preset position"""
        await self.ptz_controller.goto_preset(preset_number)
        self.logger.info(f"Camera moved to preset {preset_number}")
    
    def get_status(self):
        """Get current system status for API"""
        people_count = len(self.body_tracker.people) if hasattr(self.body_tracker, 'people') else 0
        return {
            'is_running': self.is_running,
            'is_tracking': self.is_tracking,
            'is_paused': self.is_paused,
            'primary_person_id': self.primary_person_id,
            'people_count': people_count,
            'camera_moving': self.is_camera_moving,
            'last_movement_time': self.last_movement_time
        }
    
    async def lock_primary_person(self):
        """Lock onto the primary person"""
        if self.primary_person_id:
            self.logger.info(f"Locked onto person {self.primary_person_id}")
        else:
            self.logger.warning("No primary person to lock onto")
    
    async def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        
        # Stop servers
        if self.osc_controller:
            self.osc_controller.stop()
            self.logger.info("OSC controller stopped")
        
        if self.video_capture:
            self.video_capture.stop()
            self.video_capture = None
        
        if self.debug_window:
            cv2.destroyAllWindows()
        
        if self.ptz_controller:
            self.ptz_controller.disconnect()
        
        cv2.destroyAllWindows()
        self.logger.info("PTZ tracking system cleaned up")


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)


def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ptz_tracker.log')
        ]
    )


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='PTZ Camera Tracking System')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging and window')
    parser.add_argument('--no-window', action='store_true', help='Run without debug window')
    parser.add_argument('--daemon', action='store_true', help='Run in daemon mode (no window, with API/OSC servers)')
    parser.add_argument('--home', action='store_true', help='Move camera to home position and exit')
    parser.add_argument('--preset', type=int, help='Move camera to preset position and exit')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(args.debug or config.get('system', {}).get('debug', False))
    
    logger = logging.getLogger(__name__)
    logger.info("Starting PTZ Camera Tracking System")
    
    # Create tracking system
    # In daemon mode, never show debug window
    show_debug = args.debug and not args.no_window and not args.daemon
    system = PTZTrackingSystem(config, show_debug=show_debug)
    
    try:
        await system.initialize()
        
        # Handle single commands
        if args.home:
            await system.goto_home()
            return
        elif args.preset:
            await system.goto_preset(args.preset)
            return
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received signal, shutting down...")
            system.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # In daemon mode, just keep servers alive without starting tracking
        if args.daemon:
            logger.info("Daemon mode: OSC server ready, waiting for commands...")
            system.is_running = True
            
            # Start OSC command queue processing if OSC is enabled
            osc_task = None
            if system.osc_controller and system.osc_controller.enabled:
                osc_task = asyncio.create_task(system.osc_controller.process_command_queue())
            
            # Start the main tracking loop but not actively tracking until commanded
            tracking_task = asyncio.create_task(system.run())
            
            # Lightweight daemon loop - just keep alive and monitor tasks
            try:
                while system.is_running:
                    # Check if tracking task is still running
                    if tracking_task.done():
                        # If tracking task finished, restart it
                        tracking_task = asyncio.create_task(system.run())
                    await asyncio.sleep(1.0)  # Minimal resource usage
            except KeyboardInterrupt:
                logger.info("Daemon interrupted")
            finally:
                # Clean up tasks
                if osc_task:
                    osc_task.cancel()
                    try:
                        await osc_task
                    except asyncio.CancelledError:
                        pass
                
                if tracking_task:
                    tracking_task.cancel()
                    try:
                        await tracking_task
                    except asyncio.CancelledError:
                        pass
        else:
            # Interactive mode: start tracking immediately
            await system.start_tracking()
            await system.run()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        sys.exit(0)