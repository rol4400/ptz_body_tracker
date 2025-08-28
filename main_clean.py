#!/usr/bin/env python3
"""
PTZ Camera Tracking System - Main Application (Clean Version)
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
from src.api_server import APIServer


class LatencyOptimizedCapture:
    """Video capture with minimal latency using threading"""
    
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.capture = None
        self.frame_queue = Queue(maxsize=2)
        self.latest_frame = None
        self.capture_thread = None
        self.running = False
        
    def start(self):
        """Start capture thread"""
        self.capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            return False
            
        # Optimize for low latency
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return True
    
    def _capture_loop(self):
        """Continuous capture loop that keeps only the latest frame"""
        while self.running and self.capture:
            ret, frame = self.capture.read()
            if ret:
                self.latest_frame = frame
                try:
                    while not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except:
                    pass
    
    def read(self):
        """Get the latest frame"""
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except Empty:
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
    
    def __init__(self, config: dict, daemon_mode: bool = False):
        self.config = config
        self.daemon_mode = daemon_mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.body_tracker = BodyTracker(config)
        self.ptz_controller = PTZController(config)
        
        # Video capture
        self.video_capture: Optional[LatencyOptimizedCapture] = None
        
        # Debug window (only in non-daemon mode)
        if not daemon_mode:
            self.debug_window = DebugWindow("PTZ Camera Tracker")
        else:
            self.debug_window = None
        
        # OSC controller
        self.osc_controller = OSCController(config, self)
        
        # API server
        self.api_server = APIServer(self, config.get('api', {}))
        
        # System state
        self.is_running = False
        self.is_tracking = False
        self.is_paused = False
        
        # Tracking state
        self.primary_person_id = None
        self.people_count = 0
        self.lock_status = "stopped"
        
        # Movement control
        self.last_movement_time = 0
        self.movement_cooldown = 0.1
        self.is_camera_moving = False
        self.dead_zone_width = config.get('tracking', {}).get('dead_zone_width', 0.3)
        
        # Person tracking
        self.person_selection_frames = 0
        self.min_selection_frames = 3
        self.lost_person_timeout = 30
        self.last_primary_position = None
        
        # Movement direction control
        self.current_direction = None
        self.direction_change_cooldown = 0.2
        self.last_direction_change = 0
        
    async def initialize(self):
        """Initialize all components without moving camera"""
        self.logger.info("Initializing PTZ tracking system...")
        
        # Initialize PTZ controller (no automatic positioning)
        if not await self.ptz_controller.initialize():
            raise Exception("Failed to initialize PTZ controller")
        
        # Initialize video capture
        camera_config = self.config['camera']
        rtsp_urls = [
            camera_config.get('stream_url', f"rtsp://{camera_config['ip']}:554/stream1"),
            f"rtsp://{camera_config['ip']}:554/stream0",
            f"rtsp://{camera_config['username']}:{camera_config['password']}@{camera_config['ip']}:554/stream1"
        ]
        
        self.video_capture = None
        for rtsp_url in rtsp_urls:
            self.logger.info(f"Trying RTSP URL: {rtsp_url}")
            capture = LatencyOptimizedCapture(rtsp_url)
            
            if capture.start():
                time.sleep(0.5)
                ret, test_frame = capture.read()
                if ret and test_frame is not None:
                    self.logger.info(f"Successfully connected to: {rtsp_url}")
                    self.video_capture = capture
                    break
                else:
                    capture.stop()
            else:
                capture.stop()
        
        if not self.video_capture:
            raise Exception("Failed to open camera stream")
        
        # Start OSC controller
        if not self.osc_controller.start():
            self.logger.warning("Failed to start OSC controller")
        
        # Start API server
        try:
            await self.api_server.start()
            self.logger.info("API server started successfully")
        except Exception as e:
            self.logger.warning(f"Failed to start API server: {e}")
        
        self.logger.info("PTZ tracking system initialized (no automatic positioning)")
    
    def get_status(self) -> dict:
        """Get current system status"""
        return {
            "running": self.is_running,
            "tracking": self.is_tracking,
            "paused": self.is_paused,
            "people_count": self.people_count,
            "primary_person_id": self.primary_person_id,
            "lock_status": self.lock_status,
            "camera_moving": self.is_camera_moving
        }
    
    async def start_tracking(self):
        """Start person tracking"""
        self.is_tracking = True
        self.lock_status = "unlocked"
        self.logger.info("Person tracking started")
    
    async def stop_tracking(self):
        """Stop person tracking and minimize resource usage"""
        self.is_tracking = False
        self.primary_person_id = None
        self.lock_status = "stopped"
        await self.ptz_controller.stop_movement()
        self.is_camera_moving = False
        self.current_direction = None
        self.logger.info("Person tracking stopped - minimal resource mode")
    
    async def relock_person(self):
        """Unlock current person and relock to new target"""
        if self.is_tracking:
            self.primary_person_id = None
            self.person_selection_frames = 0
            self.last_primary_position = None
            self.lock_status = "unlocked"
            self.logger.info("Relocking to new target")
        else:
            self.logger.info("Cannot relock - tracking is stopped")
    
    def update_camera_position(self, people):
        """Update camera position based on primary person"""
        if not self.is_tracking or self.is_paused:
            return
        
        current_time = time.time()
        if current_time - self.last_movement_time < self.movement_cooldown:
            return
        
        # Update people count
        self.people_count = len(people)
        
        # Find or select primary person
        primary_person = None
        if self.primary_person_id:
            for person in people:
                if person.id == self.primary_person_id:
                    primary_person = person
                    self.lost_person_timeout = 30
                    break
            
            if not primary_person:
                self.lost_person_timeout -= 1
                if self.lost_person_timeout <= 0:
                    self.logger.info(f"Lost primary person {self.primary_person_id}")
                    self.primary_person_id = None
                    self.lock_status = "unlocked"
                    self.person_selection_frames = 0
        
        if not primary_person and people:
            # Select new primary person
            if self.last_primary_position is not None:
                closest_person = None
                min_distance = float('inf')
                
                for person in people:
                    distance = ((person.center[0] - self.last_primary_position[0])**2 + 
                              (person.center[1] - self.last_primary_position[1])**2)**0.5
                    
                    if distance < min_distance and distance < 0.2:
                        min_distance = distance
                        closest_person = person
                
                if closest_person:
                    self.primary_person_id = closest_person.id
                    primary_person = closest_person
                    self.lock_status = "locked"
                    self.person_selection_frames = 0
            
            if not primary_person:
                largest_person = max(people, key=lambda p: p.size)
                
                if not hasattr(self, 'candidate_person_id') or self.candidate_person_id != largest_person.id:
                    self.candidate_person_id = largest_person.id
                    self.person_selection_frames = 1
                else:
                    self.person_selection_frames += 1
                    
                if self.person_selection_frames >= self.min_selection_frames:
                    self.primary_person_id = largest_person.id
                    primary_person = largest_person
                    self.lock_status = "locked"
                    self.person_selection_frames = 0
        
        if primary_person:
            self.last_primary_position = primary_person.center
            
            # Horizontal tracking only
            person_x = primary_person.center[0]
            
            # Safety check: stop if camera has been moving too long
            if self.is_camera_moving and current_time - self.last_movement_time > 2.0:
                asyncio.create_task(self.ptz_controller.stop_movement())
                self.is_camera_moving = False
                self.current_direction = None
                self.logger.warning("Camera moving too long - stopping for safety")
                return
            
            # Check if person is outside horizontal dead zone
            horizontal_dead_zone = not self.ptz_controller.is_in_dead_zone(person_x)
            
            if horizontal_dead_zone:
                center_x = 0.5
                offset_from_center = person_x - center_x
                
                # Only move if person is significantly off-center
                if abs(offset_from_center) > 0.05:
                    # Determine movement direction
                    desired_direction = "right" if offset_from_center > 0 else "left"
                    
                    # Check if we need to start movement or change direction
                    if not self.is_camera_moving:
                        # Start movement
                        movement_speed = 0.02  # Very slow for smooth tracking
                        
                        if desired_direction == "right":
                            asyncio.create_task(self.ptz_controller.pan_right(movement_speed))
                        else:
                            asyncio.create_task(self.ptz_controller.pan_left(movement_speed))
                        
                        self.is_camera_moving = True
                        self.current_direction = desired_direction
                        self.last_movement_time = current_time
                        self.last_direction_change = current_time
                        self.logger.info(f"Starting smooth pan {desired_direction} (person at x={person_x:.3f})")
                    
                    elif self.current_direction != desired_direction and (current_time - self.last_direction_change) > self.direction_change_cooldown:
                        # Change direction
                        asyncio.create_task(self.ptz_controller.stop_movement())
                        
                        movement_speed = 0.02
                        if desired_direction == "right":
                            asyncio.create_task(self.ptz_controller.pan_right(movement_speed))
                        else:
                            asyncio.create_task(self.ptz_controller.pan_left(movement_speed))
                        
                        self.current_direction = desired_direction
                        self.last_direction_change = current_time
                        self.logger.info(f"Changed direction to {desired_direction}")
                    
                    self.last_movement_time = current_time
                else:
                    # Person close to center - stop
                    if self.is_camera_moving:
                        asyncio.create_task(self.ptz_controller.stop_movement())
                        self.is_camera_moving = False
                        self.current_direction = None
                        self.logger.info("Stopping - person close to center")
            else:
                # Person in dead zone - stop
                if self.is_camera_moving:
                    asyncio.create_task(self.ptz_controller.stop_movement())
                    self.is_camera_moving = False
                    self.current_direction = None
                    self.logger.info("Person in dead zone - stopping")
    
    async def run(self):
        """Main tracking loop"""
        self.is_running = True
        self.logger.info("Starting PTZ tracking...")
        
        try:
            while self.is_running:
                # Minimal processing when stopped to save resources
                if not self.is_tracking:
                    await asyncio.sleep(0.1)  # Sleep longer when stopped
                    continue
                
                # Capture frame
                if self.video_capture is None:
                    self.logger.error("Video capture is None")
                    break
                    
                ret, frame = self.video_capture.read()
                if not ret or frame is None:
                    self.logger.warning("Failed to read frame from camera")
                    await asyncio.sleep(0.1)
                    continue
                
                # Detect people (only when tracking)
                people = self.body_tracker.detect_people(frame)
                
                # Update camera position
                self.update_camera_position(people)
                
                # Show debug window if enabled
                if self.debug_window and not self.daemon_mode:
                    key = self.debug_window.show_frame(frame, people, self.primary_person_id)
                    
                    action = self.debug_window.handle_key(key)
                    if action == 'quit':
                        break
                    elif action == 'pause':
                        self.is_paused = not self.is_paused
                
                # Shorter sleep when actively tracking
                await asyncio.sleep(0.03)
                
        except Exception as e:
            self.logger.error(f"Error in tracking loop: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up...")
        
        if self.is_camera_moving:
            await self.ptz_controller.stop_movement()
        
        if self.video_capture:
            self.video_capture.stop()
        
        if self.debug_window:
            self.debug_window.destroy()
        
        self.osc_controller.stop()
        
        # Stop API server
        try:
            asyncio.run(self.api_server.stop())
        except Exception as e:
            self.logger.warning(f"Error stopping API server: {e}")
        
        self.logger.info("Cleanup complete")


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def setup_logging(debug: bool = False, daemon_mode: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    
    handlers = []
    handlers.append(logging.FileHandler('ptz_tracker.log'))
    if not daemon_mode:
        handlers.append(logging.StreamHandler(sys.stdout))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='PTZ Camera Tracking System')
    parser.add_argument('--config', '-c', default='config.json', help='Configuration file path')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    parser.add_argument('--daemon', action='store_true', help='Run in daemon mode (no GUI)')
    parser.add_argument('--no-debug-window', action='store_true', help='Disable debug window')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug, args.daemon)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("Starting PTZ Camera Tracking System")
        
        # Create and initialize tracking system
        show_debug = not args.no_debug_window and not args.daemon
        tracking_system = PTZTrackingSystem(config, daemon_mode=args.daemon)
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            tracking_system.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        await tracking_system.initialize()
        
        # Start tracking if not in daemon mode (daemon waits for OSC commands)
        if not args.daemon:
            await tracking_system.start_tracking()
        else:
            logger.info("Daemon mode: Waiting for OSC commands to start tracking")
        
        await tracking_system.run()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())