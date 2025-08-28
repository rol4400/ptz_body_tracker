"""
REST API Server for PTZ Tracker
Provides HTTP endpoints for Bitfocus Companion integration
"""

import asyncio
import logging
from typing import Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import json


class APIServer:
    """HTTP REST API server for controlling PTZ tracker"""
    
    def __init__(self, tracker, config: dict):
        self.tracker = tracker
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        if config.get('enable_cors', True):
            CORS(self.app)
        
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint for Docker"""
            return jsonify({
                'status': 'healthy',
                'service': 'ptz-tracker'
            })
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """Get current tracking status"""
            try:
                status = self.tracker.get_status()
                return jsonify({
                    'success': True,
                    'data': status
                })
            except Exception as e:
                self.logger.error(f"Status endpoint error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/start', methods=['POST'])
        def start_tracking():
            """Start tracking"""
            try:
                asyncio.run_coroutine_threadsafe(
                    self.tracker.start_tracking(),
                    asyncio.get_event_loop()
                ).result(timeout=5.0)
                
                return jsonify({
                    'success': True,
                    'message': 'Tracking started'
                })
            except Exception as e:
                self.logger.error(f"Start tracking error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/stop', methods=['POST'])
        def stop_tracking():
            """Stop tracking"""
            try:
                asyncio.run_coroutine_threadsafe(
                    self.tracker.stop_tracking(),
                    asyncio.get_event_loop()
                ).result(timeout=5.0)
                
                return jsonify({
                    'success': True,
                    'message': 'Tracking stopped'
                })
            except Exception as e:
                self.logger.error(f"Stop tracking error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/lock', methods=['POST'])
        def lock_person():
            """Lock onto primary person"""
            try:
                asyncio.run_coroutine_threadsafe(
                    self.tracker.lock_primary_person(),
                    asyncio.get_event_loop()
                ).result(timeout=5.0)
                
                return jsonify({
                    'success': True,
                    'message': 'Locked onto primary person'
                })
            except Exception as e:
                self.logger.error(f"Lock person error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/preset', methods=['POST'])
        def goto_preset():
            """Go to preset position"""
            try:
                preset_number = request.json.get('preset') if request.is_json else None
                
                if preset_number:
                    asyncio.run_coroutine_threadsafe(
                        self.tracker.ptz_controller.goto_preset(preset_number),
                        asyncio.get_event_loop()
                    ).result(timeout=5.0)
                else:
                    asyncio.run_coroutine_threadsafe(
                        self.tracker.goto_preset(),
                        asyncio.get_event_loop()
                    ).result(timeout=5.0)
                
                return jsonify({
                    'success': True,
                    'message': f'Moved to preset {preset_number or "default"}'
                })
            except Exception as e:
                self.logger.error(f"Goto preset error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/pan', methods=['POST'])
        def pan_to():
            """Pan to specific angle"""
            try:
                if not request.is_json:
                    return jsonify({
                        'success': False,
                        'error': 'JSON body required'
                    }), 400
                
                angle = request.json.get('angle')
                if angle is None:
                    return jsonify({
                        'success': False,
                        'error': 'angle parameter required'
                    }), 400
                
                asyncio.run_coroutine_threadsafe(
                    self.tracker.ptz_controller.pan_to(float(angle)),
                    asyncio.get_event_loop()
                ).result(timeout=5.0)
                
                return jsonify({
                    'success': True,
                    'message': f'Panned to {angle} degrees'
                })
            except Exception as e:
                self.logger.error(f"Pan to error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/config', methods=['GET'])
        def get_config():
            """Get current configuration"""
            try:
                # Return sanitized config (without passwords)
                safe_config = self.config.copy()
                if 'camera' in safe_config:
                    safe_config['camera'] = safe_config['camera'].copy()
                    safe_config['camera'].pop('password', None)
                
                return jsonify({
                    'success': True,
                    'data': safe_config
                })
            except Exception as e:
                self.logger.error(f"Get config error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'success': False,
                'error': 'Endpoint not found'
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'success': False,
                'error': 'Internal server error'
            }), 500
    
    async def start(self):
        """Start the API server"""
        if self.is_running:
            self.logger.warning("API server already running")
            return
        
        self.logger.info(f"Starting API server on {self.config['host']}:{self.config['port']}")
        self.is_running = True
        
        def run_server():
            try:
                self.app.run(
                    host=self.config['host'],
                    port=self.config['port'],
                    debug=False,
                    use_reloader=False,
                    threaded=True
                )
            except Exception as e:
                self.logger.error(f"API server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Give server time to start
        await asyncio.sleep(1.0)
        self.logger.info("API server started successfully")
    
    async def stop(self):
        """Stop the API server"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping API server...")
        self.is_running = False
        
        # Note: Flask development server doesn't have a clean shutdown method
        # In production, you'd use a proper WSGI server like Gunicorn
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)
        
        self.logger.info("API server stopped")