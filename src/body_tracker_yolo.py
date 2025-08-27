"""
Advanced Body Detection and Tracking Module using YOLOv8 + ByteTrack
High-performance person detection and multi-object tracking
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
import time
import math
from ultralytics import YOLO
from collections import defaultdict, OrderedDict


# Simplified ByteTracker for multi-object tracking
class SimpleTracker:
    """Simple object tracker using IoU matching"""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_count = 0
    
    def update(self, detections):
        """Update tracker with new detections"""
        # Predict all existing tracks first
        for track in self.tracks:
            track.predict()
        
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, self.tracks, self.iou_threshold)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.tracks[m[1]].update(detections[m[0]])
        
        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            trk.id = self.track_count
            self.track_count += 1
            self.tracks.append(trk)
        
        # Prepare return values - include all active tracks
        i = len(self.tracks)
        ret = []
        for trk in reversed(self.tracks):
            d = trk.get_state()
            # Return tracks that are being tracked or recently started
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or trk.age <= 3):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # Remove old tracks
            if trk.time_since_update > self.max_age:
                self.tracks.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.2):
        """Assigns detections to tracked object with improved association"""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        # Calculate both IoU and center distance
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        center_dist_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            det_center_x = (det[0] + det[2]) / 2
            det_center_y = (det[1] + det[3]) / 2
            
            for t, trk in enumerate(trackers):
                trk_state = trk.get_state()
                trk_center_x = (trk_state[0] + trk_state[2]) / 2
                trk_center_y = (trk_state[1] + trk_state[3]) / 2
                
                # IoU similarity
                iou_matrix[d, t] = self._iou(det, trk_state)
                
                # Normalized center distance
                center_dist = np.sqrt((det_center_x - trk_center_x)**2 + (det_center_y - trk_center_y)**2)
                # Normalize by image diagonal (assume 1000 pixels)
                center_dist_matrix[d, t] = center_dist / 1000.0
        
        # Combined similarity: IoU + (1 - center_distance)
        similarity_matrix = 0.7 * iou_matrix + 0.3 * (1 - center_dist_matrix)
        
        if min(similarity_matrix.shape) > 0:
            a = (similarity_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self._linear_assignment(-similarity_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter out matched with low similarity
        matches = []
        for m in matched_indices:
            if similarity_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def _linear_assignment(self, cost_matrix):
        """Simple linear assignment"""
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            return np.array([[y[i], i] for i in range(len(y)) if y[i] >= 0])
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            x, y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x, y)))
    
    def _iou(self, bb_test, bb_gt):
        """Computes IOU between two bboxes in the form [x1,y1,x2,y2]"""
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o


class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes with improved parameters"""
    count = 0
    
    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0, 0],
                                              [0, 0, 0, 1, 0, 0, 0]], np.float32)
        
        self.kf.transitionMatrix = np.array([[1, 0, 0, 0, 1, 0, 0],
                                             [0, 1, 0, 0, 0, 1, 0],
                                             [0, 0, 1, 0, 0, 0, 1],
                                             [0, 0, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 0, 1]], np.float32)
        
        # Reduced process noise for smoother tracking
        self.kf.processNoiseCov = 0.005 * np.eye(7, dtype=np.float32)
        # Reduced measurement noise for more responsive tracking  
        self.kf.measurementNoiseCov = 0.05 * np.eye(4, dtype=np.float32)
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        
        self.kf.statePre = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0], np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.confidence = bbox[4] if len(bbox) > 4 else 0.8  # Store confidence
    
    def update(self, bbox):
        """Updates the state vector with observed bbox"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Update confidence if provided
        if len(bbox) > 4:
            self.confidence = bbox[4]
            
        measurement = np.array([[np.float32(bbox[0])], [np.float32(bbox[1])], 
                               [np.float32(bbox[2])], [np.float32(bbox[3])]])
        self.kf.correct(measurement)
    
    def predict(self):
        """Advances the state vector and returns the predicted bounding box estimate"""
        # Predict the next state
        self.kf.predict()
        
        # Update age and time tracking
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Get predicted position and store in history
        predicted_state = self.get_state()
        self.history.append(predicted_state.copy())
        
        # Keep history manageable
        if len(self.history) > 100:
            self.history.pop(0)
            
        return predicted_state
    
    def get_state(self):
        """Returns the current bounding box estimate from Kalman filter"""
        # Use the current state (which includes predictions)
        state = self.kf.statePost[:4].copy()
        
        # Ensure positive width and height
        if state[2] <= 0:
            state[2] = 1
        if state[3] <= 0:
            state[3] = 1
            
        return state


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
        
        self.bbox = bbox
        self.confidence = confidence
        self.center = self._calculate_center()
        self.last_seen = time.time()
        self.size = bbox[2] * bbox[3]
        self.track_history.append(self.center)
        
        # Keep only recent history
        if len(self.track_history) > 30:
            self.track_history.pop(0)
        
        # Calculate velocity
        time_delta = 0.033  # Assume ~30fps
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
        
        variance = np.var(movements) if movements else 0
        # Lower variance = higher stability
        stability = max(0, 1.0 - variance * 10)
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
        return velocity_magnitude < 0.05  # Reduced threshold for better stability


class BodyTracker:
    """Advanced body detection and tracking using YOLOv8 + ByteTrack"""
    
    def __init__(self, config: dict):
        self.config = config
        self.tracking_config = config['tracking']
        self.logger = logging.getLogger(__name__)
        
        # Initialize YOLOv8 model
        try:
            self.model = YOLO('yolov8n.pt')  # Will download if not present
            self.logger.info("YOLOv8n model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
        
        # Initialize ByteTrack
        self.byte_tracker = BYTETracker(
            frame_rate=30,
            track_thresh=self.tracking_config['confidence_threshold'],
            track_buffer=30,
            match_thresh=0.8
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
            
            # Run YOLOv8 inference
            results = self.model(frame, verbose=False, classes=[0])  # class 0 is 'person'
            
            detected_people = []
            
            if results and len(results) > 0:
                # Extract detections
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # Convert to numpy format for ByteTrack
                    detections = []
                    height, width = frame.shape[:2]
                    
                    for box in boxes:
                        # Get box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Filter by confidence
                        if conf >= self.tracking_config['confidence_threshold']:
                            # Convert to tlbr format for ByteTrack
                            detections.append([x1, y1, x2, y2, conf])
                    
                    if detections:
                        detections = np.array(detections)
                        
                        # Update ByteTracker
                        online_targets = self.byte_tracker.update(detections)
                        
                        # Convert tracked objects to Person objects
                        for track in online_targets:
                            if track.is_activated:
                                # Convert back to normalized coordinates
                                x1, y1, x2, y2 = track.tlbr
                                norm_x = x1 / width
                                norm_y = y1 / height
                                norm_w = (x2 - x1) / width
                                norm_h = (y2 - y1) / height
                                
                                # Check minimum size
                                if norm_w * norm_h >= self.tracking_config['min_detection_size']:
                                    person = Person(
                                        track_id=track.track_id,
                                        bbox=(norm_x, norm_y, norm_w, norm_h),
                                        confidence=track.score
                                    )
                                    detected_people.append(person)
                                    self.detection_count += 1
            
            # Log detection stats periodically
            if self.frame_count % 60 == 0:  # Every 2 seconds at 30fps
                self.logger.debug(f"Detection stats: {self.detection_count} people detected in {self.frame_count} frames")
            
            return detected_people
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []
    
    def update_tracking(self, detected_people: List[Person]):
        """Update tracking with new detections from ByteTrack"""
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
        """Select the primary person to track"""
        if not self.people:
            self.primary_person_id = None
            return
        
        # If current primary person is still valid and stable, keep tracking them
        if (self.primary_person_id is not None and 
            self.primary_person_id in self.people and
            self.people[self.primary_person_id].is_stable()):
            return
        
        # Select new primary person based on comprehensive scoring
        best_person = None
        best_score = -1
        
        for person in self.people.values():
            # Scoring criteria: size, confidence, stability, center position, tracking history
            size_score = min(1.0, person.size / 0.2)  # Prefer larger people
            confidence_score = person.confidence
            stability_score = person.tracking_score
            
            # Prefer people closer to center
            distance_from_center = abs(person.center[0] - 0.5)
            center_score = max(0, 1.0 - (distance_from_center * 2))
            
            # Prefer people with longer tracking history (more reliable)
            history_score = min(1.0, len(person.track_history) / 10.0)
            
            # Combined score with weights
            total_score = (size_score * 0.25 + 
                          confidence_score * 0.25 + 
                          stability_score * 0.25 + 
                          center_score * 0.15 + 
                          history_score * 0.1)
            
            if total_score > best_score:
                best_score = total_score
                best_person = person
        
        if best_person and best_person.id != self.primary_person_id:
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