"""
Simple Demo Script for PTZ Tracker Body Detection
Tests the tracking system using your webcam
"""

import cv2
import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.person_tracker import BodyTracker
from src.utils import load_config


def main():
    """Run webcam body detection demo"""
    print("PTZ Tracker - Body Detection Demo")
    print("=================================")
    print()
    
    # Load configuration
    try:
        config = load_config("config.json")
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Could not load configuration: {e}")
        return
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize body tracker
    print("Initializing body detection...")
    tracker = BodyTracker(config)
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Could not open webcam")
        return
    
    print("✓ Webcam opened successfully")
    print()
    print("Demo Controls:")
    print("- 'q' to quit")
    print("- 'l' to lock onto primary person")
    print("- 's' to show/hide detection boxes")
    print("- Space to pause/resume")
    print()
    
    # Demo variables
    frame_count = 0
    start_time = time.time()
    show_detections = True
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Detect people
                people = tracker.detect_people(frame)
                tracker.update_tracking(people)
                
                # Get primary person position
                primary_position = tracker.get_primary_person_position()
                
                # Draw detection results
                if show_detections:
                    # Temporarily enable debug mode for visualization
                    original_debug = config.get('system', {}).get('debug', False)
                    config.setdefault('system', {})['debug'] = True
                    
                    annotated_frame = tracker.draw_detections(frame)
                    
                    # Restore original debug setting
                    config['system']['debug'] = original_debug
                else:
                    annotated_frame = frame
                
                # Add status overlay
                status_text = f"People: {tracker.get_people_count()}"
                if tracker.primary_person_id is not None:
                    status_text += f" | Primary: {tracker.primary_person_id}"
                    if primary_position:
                        status_text += f" | Pos: ({primary_position[0]:.2f}, {primary_position[1]:.2f})"
                
                cv2.putText(annotated_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add pan position indicator (simulated)
                if primary_position:
                    frame_width = annotated_frame.shape[1]
                    pan_pixel = int(primary_position[0] * frame_width)
                    
                    # Draw dead zone
                    dead_zone_width = config['tracking']['dead_zone_width']
                    dead_zone_left = int((0.5 - dead_zone_width/2) * frame_width)
                    dead_zone_right = int((0.5 + dead_zone_width/2) * frame_width)
                    
                    cv2.rectangle(annotated_frame, (dead_zone_left, 0), 
                                 (dead_zone_right, 20), (0, 255, 255), -1)
                    cv2.putText(annotated_frame, "DEAD ZONE", (dead_zone_left + 5, 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
                    # Draw pan indicator
                    cv2.line(annotated_frame, (pan_pixel, 0), (pan_pixel, 30), (255, 0, 0), 3)
                
                # Show frame
                cv2.imshow('PTZ Tracker Demo', annotated_frame)
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"FPS: {fps:.1f} | People: {tracker.get_people_count()} | "
                          f"Primary: {tracker.primary_person_id or 'None'}")
            
            # Handle key presses
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('l'):
                if tracker.lock_primary_person():
                    print("✓ Locked onto primary person")
                else:
                    print("✗ No person to lock onto")
            elif key == ord('s'):
                show_detections = not show_detections
                print(f"Detection boxes: {'ON' if show_detections else 'OFF'}")
            elif key == ord(' '):
                paused = not paused
                print(f"Demo: {'PAUSED' if paused else 'RESUMED'}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Demo completed")


if __name__ == "__main__":
    main()