"""
Simple Object Tracking Module
Contains SimpleTracker and KalmanBoxTracker for multi-object tracking
"""

import cv2
import numpy as np
from typing import Any


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