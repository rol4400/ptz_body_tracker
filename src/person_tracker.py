"""
Advanced Person Detection and Tracking Module using YOLOv8
High-performance person detection and multi-object tracking
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
import time
import math
from ultralytics import YOLO
from .simple_tracker import SimpleTracker


class Person:
    """Represents a detected and tracked person"""
    
    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float], confidence: float):
        self.id = track_id
        self.bbox = bbox  # (x, y, width, height) normalized
        self.confidence = confidence
        self.center = self._calculate_center()
        self.last_seen = time.time()
        self.tracking_score = confidence
        self.velocity = (0.0, 0.0)
        self.size = bbox[2] * bbox[3]  # width * height
        self.track_history = []
        
    def _calculate_center(self) -> Tuple[float, float]:
        """Calculate center point of person"""
        return (self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2)
    
    def update_position(self, bbox: Tuple[float, float, float, float], confidence: float):
        """Update person's position and calculate velocity"""
        old_center = self.center
        old_bbox = self.bbox
        
        self.bbox = bbox
        self.confidence = confidence
        self.center = self._calculate_center()
        self.last_seen = time.time()
        self.size = bbox[2] * bbox[3]
        self.track_history.append(self.center)
        
        # Keep only recent history
        if len(self.track_history) > 30:
            self.track_history.pop(0)
        
        # Calculate velocity based on actual position change
        time_delta = 0.033  # Assume ~30fps
        if old_center != self.center:  # Only update if position actually changed
            self.velocity = (
                (self.center[0] - old_center[0]) / time_delta,
                (self.center[1] - old_center[1]) / time_delta
            )
        
        # Update tracking score based on confidence and stability
        stability = self._calculate_stability()
        self.tracking_score = confidence * 0.7 + stability * 0.3
    
    def _calculate_stability(self) -> float:
        """Calculate tracking stability based on movement history"""
        if len(self.track_history) < 5:
            return 0.5
        
        # Calculate movement variance
        recent_moves = self.track_history[-5:]
        if len(recent_moves) < 2:
            return 0.5
        
        movements = []
        for i in range(1, len(recent_moves)):
            dx = recent_moves[i][0] - recent_moves[i-1][0]
            dy = recent_moves[i][1] - recent_moves[i-1][1]
            movements.append(math.sqrt(dx*dx + dy*dy))
        
        variance = float(np.var(movements)) if movements else 0.0
        stability = max(0.0, 1.0 - variance * 10.0)
        return min(1.0, stability)
    
    def distance_to(self, other_center: Tuple[float, float]) -> float:
        """Calculate distance to another point"""
        return math.sqrt(
            (self.center[0] - other_center[0]) ** 2 + 
            (self.center[1] - other_center[1]) ** 2
        )
    
    def is_stable(self) -> bool:
        """Check if person is moving slowly (stable for tracking)"""
        velocity_magnitude = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        return velocity_magnitude < 0.05


class BodyTracker:
    """Advanced body detection and tracking using YOLOv8 + SimpleTracker"""
    
    def __init__(self, config: dict):
        self.config = config
        self.tracking_config = config['tracking']
        self.logger = logging.getLogger(__name__)
        
        # Initialize YOLOv8 model with GPU support if available
        try:
            self.model = YOLO('yolov8n.pt')  # Will download if not present
            
            # Check for CUDA availability and set device
            import torch
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.model.to('cuda')
                self.logger.info(f"YOLOv8n model loaded with GPU acceleration (CUDA device count: {torch.cuda.device_count()})")
            else:
                self.device = 'cpu'
                self.logger.info("YOLOv8n model loaded on CPU (CUDA not available)")
                
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
        
        # Initialize Simple Tracker with more permissive settings
        self.tracker = SimpleTracker(
            max_age=60,       # Keep tracks longer (2 seconds at 30fps)
            min_hits=1,       # Start tracking immediately 
            iou_threshold=0.2 # More permissive IoU for better association
        )
        
        # Tracking state
        self.people: Dict[int, Person] = {}
        self.primary_person_id: Optional[int] = None
        self.last_detection_time = time.time()
        
        # Movement smoothing
        self.position_history = []
        self.smoothed_position = (0.5, 0.5)  # Center screen default
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        
    def detect_people(self, frame: np.ndarray) -> List[Person]:
        """Detect people in the current frame using YOLOv8"""
        try:
            self.frame_count += 1
            
            # Always update tracker (even with empty detections for prediction)
            detections = []
            
            # Run YOLOv8 inference with device specification
            results = self.model(frame, verbose=False, classes=[0], device=self.device)  # class 0 is 'person'
            
            if results and len(results) > 0:
                # Extract detections
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    height, width = frame.shape[:2]
                    
                    for box in boxes:
                        # Get box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Use lower threshold for tracking (more permissive)
                        tracking_threshold = max(0.3, self.tracking_config['confidence_threshold'] - 0.2)
                        if conf >= tracking_threshold:
                            detections.append([x1, y1, x2, y2, conf])  # Include confidence
            
            # Always update tracker (handles prediction when no detections)
            detections_array = np.array(detections) if detections else np.empty((0, 5))
            tracked_objects = self.tracker.update(detections_array)
            
            detected_people = []
            
            # Convert tracked objects to Person objects
            if len(tracked_objects) > 0:
                height, width = frame.shape[:2]
                
                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = obj[0], obj[1], obj[2], obj[3], int(obj[4])
                    
                    # Convert to normalized coordinates
                    norm_x = x1 / width
                    norm_y = y1 / height
                    norm_w = (x2 - x1) / width
                    norm_h = (y2 - y1) / height
                    
                    # Check minimum size
                    if norm_w * norm_h >= self.tracking_config['min_detection_size']:
                        # Find matching detection to get confidence
                        confidence = 0.7  # Default confidence for tracked objects
                        
                        # Try to match with original detections for better confidence
                        for i, det in enumerate(detections):
                            dx1, dy1, dx2, dy2 = det[0], det[1], det[2], det[3]
                            if abs(dx1 - x1) < 20 and abs(dy1 - y1) < 20:  # Close match
                                confidence = det[4] if len(det) > 4 else 0.7
                                break
                        
                        person = Person(
                            track_id=track_id,
                            bbox=(norm_x, norm_y, norm_w, norm_h),
                            confidence=confidence
                        )
                        detected_people.append(person)
                        self.detection_count += 1
            
            # Log detection stats periodically
            if self.frame_count % 60 == 0:  # Every 2 seconds at 30fps
                self.logger.info(f"Detection stats: {self.detection_count} people detected in {self.frame_count} frames")
            
            return detected_people
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []
    
    def update_tracking(self, detected_people: List[Person]):
        """Update tracking with new detections"""
        current_time = time.time()
        
        # Update people dictionary with new detections
        new_people = {}
        for person in detected_people:
            if person.id in self.people:
                # Update existing person
                existing_person = self.people[person.id]
                existing_person.update_position(person.bbox, person.confidence)
                new_people[person.id] = existing_person
            else:
                # New person detected
                new_people[person.id] = person
        
        # Remove people that haven't been seen recently
        for person_id, person in new_people.items():
            if current_time - person.last_seen <= self.tracking_config['lost_tracking_timeout']:
                self.people[person_id] = person
        
        # Clean up old tracks
        self.people = {pid: person for pid, person in self.people.items() 
                      if current_time - person.last_seen <= self.tracking_config['lost_tracking_timeout']}
        
        self.last_detection_time = current_time
        
        # Update primary person selection
        self._update_primary_person()
    
    def _update_primary_person(self):
        """Select the primary person to track with improved stability"""
        if not self.people:
            self.primary_person_id = None
            return
        
        # If current primary person is still valid, be conservative about switching
        if (self.primary_person_id is not None and 
            self.primary_person_id in self.people):
            current_primary = self.people[self.primary_person_id]
            # Only switch if current primary has very low confidence or stopped being detected
            if current_primary.confidence > 0.4 and current_primary.tracking_score > 0.3:
                return
        
        # Select new primary person based on comprehensive scoring
        best_person = None
        best_score = -1
        
        for person in self.people.values():
            # Scoring criteria: size, confidence, stability, center position
            size_score = min(1.0, person.size / 0.15)  # Slightly lower size requirement
            confidence_score = person.confidence
            stability_score = person.tracking_score
            
            # Prefer people closer to center
            distance_from_center = abs(person.center[0] - 0.5)
            center_score = max(0, 1.0 - (distance_from_center * 1.5))  # Less center bias
            
            # Prefer tracks with longer history (more reliable)
            history_score = min(1.0, len(person.track_history) / 10.0)
            
            # Combined score with adjusted weights
            total_score = (size_score * 0.25 + 
                          confidence_score * 0.3 + 
                          stability_score * 0.25 + 
                          center_score * 0.1 + 
                          history_score * 0.1)
            
            if total_score > best_score:
                best_score = total_score
                best_person = person
        
        # Only switch if significantly better or no current primary
        if best_person and (self.primary_person_id is None or 
                           self.primary_person_id not in self.people or
                           best_score > 0.6):  # Require minimum quality
            if best_person.id != self.primary_person_id:
                self.primary_person_id = best_person.id
                self.logger.info(f"Selected primary person: {self.primary_person_id} (score: {best_score:.3f})")
    
    def get_primary_person_position(self) -> Optional[Tuple[float, float]]:
        """Get the smoothed position of the primary person"""
        if (self.primary_person_id is None or 
            self.primary_person_id not in self.people):
            return None
        
        primary_person = self.people[self.primary_person_id]
        current_position = primary_person.center
        
        # Apply smoothing
        smoothing_factor = self.tracking_config['smoothing_factor']
        self.smoothed_position = (
            self.smoothed_position[0] * (1 - smoothing_factor) + current_position[0] * smoothing_factor,
            self.smoothed_position[1] * (1 - smoothing_factor) + current_position[1] * smoothing_factor
        )
        
        return self.smoothed_position
    
    def lock_primary_person(self) -> bool:
        """Lock onto the most prominent person currently visible"""
        if not self.people:
            return False
        
        # Find the best person based on comprehensive scoring
        best_person = max(
            self.people.values(),
            key=lambda p: p.size * p.confidence * p.tracking_score
        )
        
        self.primary_person_id = best_person.id
        self.logger.info(f"Locked onto person: {self.primary_person_id}")
        return True
    
    def is_tracking_lost(self) -> bool:
        """Check if tracking is lost"""
        current_time = time.time()
        return (not self.people or 
                current_time - self.last_detection_time > self.tracking_config['lost_tracking_timeout'])
    
    def get_people_count(self) -> int:
        """Get current number of tracked people"""
        return len(self.people)
    
    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection results on frame (for debugging)"""
        if not self.config.get('system', {}).get('debug', False):
            return frame
        
        annotated_frame = frame.copy()
        
        for person in self.people.values():
            # Draw bounding box
            h, w = frame.shape[:2]
            x1 = int(person.bbox[0] * w)
            y1 = int(person.bbox[1] * h)
            x2 = int((person.bbox[0] + person.bbox[2]) * w)
            y2 = int((person.bbox[1] + person.bbox[3]) * h)
            
            # Color coding: Green for primary, Blue for others
            color = (0, 255, 0) if person.id == self.primary_person_id else (255, 0, 0)
            thickness = 3 if person.id == self.primary_person_id else 2
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point
            center_x = int(person.center[0] * w)
            center_y = int(person.center[1] * h)
            cv2.circle(annotated_frame, (center_x, center_y), 8, color, -1)
            
            # Draw tracking trail
            if len(person.track_history) > 1:
                trail_points = [(int(pos[0] * w), int(pos[1] * h)) for pos in person.track_history[-10:]]
                for i in range(1, len(trail_points)):
                    alpha = i / len(trail_points)
                    trail_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                    cv2.line(annotated_frame, trail_points[i-1], trail_points[i], trail_color, 2)
            
            # Draw ID and metrics
            label = f"ID:{person.id} C:{person.confidence:.2f} S:{person.tracking_score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(annotated_frame, (x1, y1-label_size[1]-10), 
                         (x1+label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw frame stats
        stats_text = f"Frame: {self.frame_count} | People: {len(self.people)} | Primary: {self.primary_person_id or 'None'}"
        cv2.putText(annotated_frame, stats_text, (10, annotated_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated_frame