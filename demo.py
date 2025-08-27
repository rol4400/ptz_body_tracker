#!/usr/bin/env python3
"""
PTZ Camera Tracking System - Demo (Webcam Version)
Demonstrates person tracking using laptop webcam without PTZ control
"""

import cv2
import asyncio
import logging
import threading
import json
from pathlib import Path
from typing import List

# Local imports
from src.person_tracker import BodyTracker
from src.debug_window import DebugWindow


class WebcamTrackingDemo:
    """Webcam-based person tracking demo"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the webcam tracking system"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.body_tracker = BodyTracker(self.config)
        self.debug_window = DebugWindow("PTZ Tracker Demo - Webcam")
        
        # Video capture
        self.video_capture: cv2.VideoCapture = None  # type: ignore
        
        # Tracking state
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = False
        self.is_paused = False
        
        # Primary person tracking
        self.primary_person_id = None
        self.dead_zone_width = self.config['tracking']['dead_zone_width']
        
        self.logger.info("PTZ Tracker Demo initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('ptz_tracker_demo.log')
            ]
        )
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize webcam (camera index 0)
            self.logger.info("Connecting to webcam...")
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                raise RuntimeError("Failed to connect to webcam")
            
            # Set webcam properties for better performance
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.logger.info("✓ Connected to webcam")
            self.logger.info("✓ Body tracker initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        
        if self.video_capture:
            self.video_capture.release()
        
        cv2.destroyAllWindows()
        self.logger.info("Demo cleanup completed")
    
    def update_tracking(self, people: List):
        """Update tracking logic (simplified for demo)"""
        if not people:
            if self.primary_person_id is not None:
                self.logger.info("Lost primary person")
                self.primary_person_id = None
            return
        
        # Find current primary person
        primary_person = None
        if self.primary_person_id:
            primary_person = next((p for p in people if p.id == self.primary_person_id), None)
        
        if not primary_person and people:
            # Select largest person as primary
            primary_person = max(people, key=lambda p: p.size)
            self.primary_person_id = primary_person.id
            self.logger.info(f"Selected primary person: {self.primary_person_id}")
        
        if primary_person:
            # Calculate if person is in center or needs tracking (demo purposes)
            person_x = primary_person.center[0]  # Normalized X position (0-1)
            
            center_x = 0.5
            dead_zone_left = center_x - (self.dead_zone_width / 2)
            dead_zone_right = center_x + (self.dead_zone_width / 2)
            
            if person_x < dead_zone_left:
                self.logger.debug(f"Person on left side (x={person_x:.2f}) - would pan left")
            elif person_x > dead_zone_right:
                self.logger.debug(f"Person on right side (x={person_x:.2f}) - would pan right")
            else:
                self.logger.debug(f"Person in center dead zone (x={person_x:.2f}) - no movement needed")
    
    async def run(self):
        """Main tracking loop"""
        self.is_running = True
        self.logger.info("Starting webcam demo...")
        
        try:
            while self.is_running:
                # Capture frame
                ret, frame = self.video_capture.read()
                if not ret:
                    self.logger.warning("Failed to read frame from webcam")
                    await asyncio.sleep(0.1)
                    continue
                
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                if not self.is_paused:
                    # Detect people
                    people = self.body_tracker.detect_people(frame)
                    
                    # Update tracking logic (demo only - no actual camera movement)
                    self.update_tracking(people)
                    
                    # Update debug window
                    key = self.debug_window.show_frame(frame, people)
                else:
                    # Show paused frame
                    key = self.debug_window.show_frame(frame)
                
                # Handle debug window events
                if key == ord('q'):
                    self.logger.info("Quit requested")
                    break
                elif key == ord(' '):
                    self.is_paused = not self.is_paused
                    self.logger.info(f"Demo {'paused' if self.is_paused else 'resumed'}")
                elif key == ord('r'):
                    self.primary_person_id = None
                    self.logger.info("Reset primary person tracking")
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in demo loop: {e}")
        finally:
            self.cleanup()


async def main():
    """Main entry point"""
    # Check if config file exists
    config_path = Path("config.json")
    if not config_path.exists():
        print("Error: config.json not found. Please ensure configuration file exists.")
        return 1
    
    # Create and run demo
    demo = WebcamTrackingDemo()
    
    print("PTZ Camera Tracking Demo - Webcam Version")
    print("==========================================")
    print("This demo shows person tracking using your webcam.")
    print("No actual PTZ camera control is performed.")
    print("")
    print("Controls:")
    print("  SPACE - Pause/Resume tracking")
    print("  R     - Reset primary person tracking")
    print("  Q     - Quit demo")
    print("")
    
    if await demo.initialize():
        print("Demo initialized successfully. Starting tracking...")
        await demo.run()
        return 0
    else:
        print("Failed to initialize demo")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)