"""
Utility functions for PTZ Tracker
"""

import json
import logging
import time
import math
from typing import Dict, Any, Tuple


def setup_logging(level: str = "INFO", log_file: str = "ptz_tracker.log"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON file"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save configuration: {e}")


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180] range"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def smooth_value(current: float, target: float, smoothing_factor: float) -> float:
    """Apply smoothing to a value"""
    return current * (1 - smoothing_factor) + target * smoothing_factor


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value to range"""
    return max(min_value, min(max_value, value))


def is_point_in_rect(point: Tuple[float, float], rect: Tuple[float, float, float, float]) -> bool:
    """Check if point is inside rectangle"""
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self, name: str = "monitor"):
        self.name = name
        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
    
    def update(self):
        """Update performance metrics"""
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS every second
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps
    
    def get_uptime(self) -> float:
        """Get uptime in seconds"""
        return time.time() - self.start_time


class MovingAverage:
    """Calculate moving average of values"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.values = []
    
    def add(self, value: float):
        """Add new value"""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    def get_average(self) -> float:
        """Get current average"""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def reset(self):
        """Reset the average"""
        self.values.clear()


class RateLimiter:
    """Rate limiter for controlling operation frequency"""
    
    def __init__(self, max_rate: float):
        self.max_rate = max_rate
        self.min_interval = 1.0 / max_rate if max_rate > 0 else 0
        self.last_execution = 0
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        current_time = time.time()
        if current_time - self.last_execution >= self.min_interval:
            self.last_execution = current_time
            return True
        return False
    
    def time_until_next(self) -> float:
        """Get time until next execution is allowed"""
        current_time = time.time()
        elapsed = current_time - self.last_execution
        return max(0, self.min_interval - elapsed)