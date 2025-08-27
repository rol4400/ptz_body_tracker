"""
Main PTZ Tracker Module
Coordinates video capture, body tracking, and PTZ camera control
"""

import cv2
import asyncio
import logging
import time
import numpy as np
from typing import Optional, Tuple
import threading

from .camera_controller import PTZController
from .person_tracker import BodyTracker


class PTZTracker:
    """Main tracking system that coordinates all components"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.ptz_controller = PTZController(config)
        self.body_tracker = BodyTracker(config)
        
        # Video capture
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.frame_thread: Optional[threading.Thread] = None
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        # Tracking state
        self.is_tracking = False
        self.is_running = False
        self.last_movement_time = 0
        self.movement_cooldown = 0.1  # Minimum time between movements
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    async def initialize(self) -> bool:
        """Initialize all components"""
        self.logger.info("Initializing PTZ Tracker...")
        
        # Initialize PTZ controller
        if not await self.ptz_controller.initialize():
            self.logger.error("Failed to initialize PTZ controller")
            return False
        
        # Initialize video capture
        if not self._initialize_video_capture():
            self.logger.error("Failed to initialize video capture")
            return False
        
        self.logger.info("PTZ Tracker initialized successfully")
        return True
    
    def _initialize_video_capture(self) -> bool:
        """Initialize video capture from camera stream"""
        try:
            stream_url = self.config['camera']['stream_url']
            self.video_capture = cv2.VideoCapture(stream_url)
            
            if not self.video_capture.isOpened():
                # Fallback to camera index if RTSP fails
                self.video_capture = cv2.VideoCapture(0)
            
            if self.video_capture.isOpened():
                # Set capture properties for better performance
                self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.video_capture.set(cv2.CAP_PROP_FPS, 30)
                self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                self.logger.info(f"Video capture initialized: {stream_url}")
                return True
            else:
                self.logger.error("Could not open video capture")
                return False
                
        except Exception as e:
            self.logger.error(f"Video capture initialization error: {e}")
            return False
    
    def _capture_frames(self):
        """Capture frames in separate thread"""
        while self.is_running:
            try:
                ret, frame = self.video_capture.read()
                if ret:
                    with self.frame_lock:
                        self.current_frame = frame
                else:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Frame capture error: {e}")
                time.sleep(0.1)
    
    async def start_tracking(self):
        """Start the tracking process"""
        if self.is_tracking:
            self.logger.warning("Tracking already started")
            return
        
        self.logger.info("Starting tracking...")
        self.is_tracking = True
        
        # Start frame capture thread if not running
        if not self.is_running:
            self.is_running = True
            self.frame_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.frame_thread.start()
    
    async def stop_tracking(self):
        """Stop the tracking process"""
        if not self.is_tracking:
            return
        
        self.logger.info("Stopping tracking...")
        self.is_tracking = False
        
        # Stop PTZ movement
        await self.ptz_controller.stop_movement()
    
    async def lock_primary_person(self):
        """Lock onto the most prominent person currently visible"""
        if not self.current_frame is None:
            # Process current frame to detect people
            people = self.body_tracker.detect_people(self.current_frame)
            self.body_tracker.update_tracking(people)
            
            if self.body_tracker.lock_primary_person():
                self.logger.info("Successfully locked onto primary person")
            else:
                self.logger.warning("No people detected to lock onto")
        else:
            self.logger.warning("No current frame available for locking")
    
    async def goto_preset(self):
        """Move camera to default preset position"""
        await self.ptz_controller.goto_preset()
    
    async def run(self):
        """Main tracking loop"""
        self.logger.info("Starting main tracking loop...")
        
        # Start frame capture if not already running
        if not self.is_running:
            self.is_running = True
            self.frame_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.frame_thread.start()
        
        try:
            while self.is_running:
                await self._process_frame()
                await asyncio.sleep(1.0 / self.config['system']['fps_limit'])
                
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
        finally:
            await self.cleanup()
    
    async def _process_frame(self):
        """Process a single frame for tracking"""
        if not self.is_tracking or self.current_frame is None:
            return
        
        try:
            # Get current frame
            with self.frame_lock:
                frame = self.current_frame.copy()
            
            # Skip frames for performance if configured
            frame_skip = self.config['system'].get('frame_skip', 1)
            if self.fps_counter % frame_skip != 0:
                self.fps_counter += 1
                return
            
            # Detect people in frame
            people = self.body_tracker.detect_people(frame)
            self.body_tracker.update_tracking(people)
            
            # Get primary person position
            primary_position = self.body_tracker.get_primary_person_position()
            
            if primary_position:
                await self._handle_tracking(primary_position)
            else:
                await self._handle_lost_tracking()
            
            # Update FPS counter
            self.fps_counter += 1
            if self.fps_counter % 30 == 0:
                elapsed = time.time() - self.fps_start_time
                fps = 30 / elapsed
                self.logger.debug(f"Processing FPS: {fps:.1f}")
                self.fps_start_time = time.time()
                
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
    
    async def _handle_tracking(self, position: Tuple[float, float]):
        """Handle tracking when person is detected"""
        x_pos, y_pos = position
        
        # Check if position is outside dead zone
        if not self.ptz_controller.is_in_dead_zone(x_pos):
            current_time = time.time()
            
            # Respect movement cooldown
            if current_time - self.last_movement_time >= self.movement_cooldown:
                # Calculate target pan angle
                target_pan = self.ptz_controller.calculate_pan_for_position(x_pos)
                
                # Move camera
                await self.ptz_controller.pan_to(target_pan)
                self.last_movement_time = current_time
                
                self.logger.debug(f"Tracking: pos=({x_pos:.3f}, {y_pos:.3f}), pan={target_pan:.1f}")
    
    async def _handle_lost_tracking(self):
        """Handle when tracking is lost"""
        if self.body_tracker.is_tracking_lost():
            self.logger.warning("Tracking lost - returning to preset")
            await self.goto_preset()
            
            # Reset tracking state
            self.body_tracker.primary_person_id = None
    
    async def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up PTZ Tracker...")
        
        self.is_running = False
        self.is_tracking = False
        
        # Stop PTZ movement
        await self.ptz_controller.stop_movement()
        
        # Release video capture
        if self.video_capture:
            self.video_capture.release()
        
        # Wait for frame thread to finish
        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join(timeout=1.0)
        
        self.logger.info("Cleanup complete")
    
    def get_status(self) -> dict:
        """Get current tracking status"""
        return {
            'is_tracking': self.is_tracking,
            'is_running': self.is_running,
            'people_count': self.body_tracker.get_people_count(),
            'primary_person_id': self.body_tracker.primary_person_id,
            'current_pan': self.ptz_controller.current_pan,
            'tracking_lost': self.body_tracker.is_tracking_lost()
        }