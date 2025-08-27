#!/usr/bin/env python3
"""
Debug Window Module
Provides a shared debug window for displaying tracking information
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time


class DebugWindow:
    """Debug window for displaying tracking information"""
    
    def __init__(self, window_name: str = "PTZ Tracker Debug", show_fps: bool = True):
        self.window_name = window_name
        self.show_fps = show_fps
        self.show_boxes = True
        self.show_ids = True
        self.show_confidence = True
        
        # FPS calculation
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Colors for different person IDs
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
        ]
    
    def update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_color_for_id(self, person_id: int) -> Tuple[int, int, int]:
        """Get consistent color for person ID"""
        return self.colors[person_id % len(self.colors)]
    
    def draw_person_box(self, frame: np.ndarray, person: Any, is_primary: bool = False):
        """Draw bounding box and info for a person"""
        if not self.show_boxes:
            return
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Convert normalized bbox to pixel coordinates
        x, y, width, height = person.bbox
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + width) * w)
        y2 = int((y + height) * h)
        
        # Get color (primary person gets special color)
        if is_primary:
            color = (0, 255, 0)  # Bright green for primary
            thickness = 3
        else:
            color = self.get_color_for_id(person.id)
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw center point
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        # Prepare label text
        label_parts = []
        if self.show_ids:
            label_parts.append(f"ID:{person.id}")
        if self.show_confidence:
            label_parts.append(f"{person.confidence:.2f}")
        if is_primary:
            label_parts.append("PRIMARY")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Position label above the box
            label_x = x1
            label_y = y1 - 10
            if label_y < text_height + 5:
                label_y = y2 + text_height + 5
            
            # Draw label background
            cv2.rectangle(frame, 
                         (label_x, label_y - text_height - 5),
                         (label_x + text_width + 5, label_y + baseline),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (label_x + 2, label_y - 2),
                       font, font_scale, (255, 255, 255), font_thickness)
    
    def draw_tracking_info(self, frame: np.ndarray, people: List[Any], primary_person_id: Optional[int] = None):
        """Draw all tracking information on frame"""
        # Draw all people
        for person in people:
            is_primary = primary_person_id is not None and person.id == primary_person_id
            self.draw_person_box(frame, person, is_primary)
        
        # Draw status information
        self.draw_status_info(frame, len(people), primary_person_id)
    
    def draw_status_info(self, frame: np.ndarray, people_count: int, primary_person_id: Optional[int] = None):
        """Draw status information overlay"""
        h, w = frame.shape[:2]
        
        # Status text
        status_lines = []
        if self.show_fps:
            status_lines.append(f"FPS: {self.current_fps:.1f}")
        status_lines.append(f"People: {people_count}")
        if primary_person_id is not None:
            status_lines.append(f"Primary: {primary_person_id}")
        else:
            status_lines.append("Primary: None")
        
        # Draw status background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        line_height = 30
        
        max_width = 0
        for line in status_lines:
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            max_width = max(max_width, text_width)
        
        status_height = len(status_lines) * line_height + 10
        cv2.rectangle(frame, (10, 10), (max_width + 20, status_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (max_width + 20, status_height), (255, 255, 255), 2)
        
        # Draw status text
        for i, line in enumerate(status_lines):
            y_pos = 35 + i * line_height
            cv2.putText(frame, line, (15, y_pos), font, font_scale, (255, 255, 255), font_thickness)
    
    def draw_controls_help(self, frame: np.ndarray):
        """Draw control help text"""
        h, w = frame.shape[:2]
        
        help_lines = [
            "'q' - Quit",
            "'l' - Lock primary person",
            "'s' - Toggle detection boxes",
            "'h' - Toggle this help",
            "Space - Pause/Resume"
        ]
        
        # Position help at bottom right
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        line_height = 20
        
        max_width = 0
        for line in help_lines:
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            max_width = max(max_width, text_width)
        
        help_height = len(help_lines) * line_height + 10
        start_x = w - max_width - 20
        start_y = h - help_height - 10
        
        # Draw help background
        cv2.rectangle(frame, (start_x, start_y), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (start_x, start_y), (w - 10, h - 10), (255, 255, 255), 1)
        
        # Draw help text
        for i, line in enumerate(help_lines):
            y_pos = start_y + 20 + i * line_height
            cv2.putText(frame, line, (start_x + 5, y_pos), font, font_scale, (255, 255, 255), font_thickness)
    
    def show_frame(self, frame: np.ndarray, people: List[Any] = None, primary_person_id: Optional[int] = None, show_help: bool = False):
        """Display frame with tracking information"""
        display_frame = frame.copy()
        
        # Update FPS
        self.update_fps()
        
        # Draw tracking info if people are provided
        if people is not None:
            self.draw_tracking_info(display_frame, people, primary_person_id)
        
        # Draw controls help if requested
        if show_help:
            self.draw_controls_help(display_frame)
        
        # Ensure window displays full frame without cropping
        # Create window with resizable property to maintain aspect ratio
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        
        # Show the frame at full resolution
        cv2.imshow(self.window_name, display_frame)
        
        return cv2.waitKey(1) & 0xFF
    
    def handle_key(self, key: int) -> str:
        """Handle key press and return action"""
        if key == ord('q'):
            return 'quit'
        elif key == ord('l'):
            return 'lock'
        elif key == ord('s'):
            self.show_boxes = not self.show_boxes
            return 'toggle_boxes'
        elif key == ord('h'):
            return 'toggle_help'
        elif key == 32:  # Space
            return 'pause'
        return 'none'
    
    def destroy(self):
        """Clean up the window"""
        cv2.destroyWindow(self.window_name)