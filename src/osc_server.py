"""
OSC Server for PTZ Tracker
Provides OSC endpoints for real-time control
"""

import asyncio
import logging
from typing import Optional
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
import threading


class OSCServer:
    """OSC server for controlling PTZ tracker"""
    
    def __init__(self, tracker, config: dict):
        self.tracker = tracker
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.server: Optional[ThreadingOSCUDPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        self._setup_dispatcher()
    
    def _setup_dispatcher(self):
        """Setup OSC message dispatcher"""
        self.dispatcher = Dispatcher()
        
        # Register handlers
        self.dispatcher.map("/ptz/start", self._handle_start)
        self.dispatcher.map("/ptz/stop", self._handle_stop)
        self.dispatcher.map("/ptz/lock", self._handle_lock)
        self.dispatcher.map("/ptz/preset", self._handle_preset)
        self.dispatcher.map("/ptz/pan", self._handle_pan)
        self.dispatcher.map("/ptz/status", self._handle_status)
        
        # Default handler for unrecognized messages
        self.dispatcher.set_default_handler(self._default_handler)
    
    def _handle_start(self, unused_addr, *args):
        """Handle start tracking command"""
        try:
            asyncio.run_coroutine_threadsafe(
                self.tracker.start_tracking(),
                asyncio.get_event_loop()
            )
            self.logger.info("OSC: Start tracking command received")
        except Exception as e:
            self.logger.error(f"OSC start tracking error: {e}")
    
    def _handle_stop(self, unused_addr, *args):
        """Handle stop tracking command"""
        try:
            asyncio.run_coroutine_threadsafe(
                self.tracker.stop_tracking(),
                asyncio.get_event_loop()
            )
            self.logger.info("OSC: Stop tracking command received")
        except Exception as e:
            self.logger.error(f"OSC stop tracking error: {e}")
    
    def _handle_lock(self, unused_addr, *args):
        """Handle lock person command"""
        try:
            asyncio.run_coroutine_threadsafe(
                self.tracker.lock_primary_person(),
                asyncio.get_event_loop()
            )
            self.logger.info("OSC: Lock person command received")
        except Exception as e:
            self.logger.error(f"OSC lock person error: {e}")
    
    def _handle_preset(self, unused_addr, *args):
        """Handle goto preset command"""
        try:
            preset_number = args[0] if args else None
            
            if preset_number:
                asyncio.run_coroutine_threadsafe(
                    self.tracker.ptz_controller.goto_preset(int(preset_number)),
                    asyncio.get_event_loop()
                )
            else:
                asyncio.run_coroutine_threadsafe(
                    self.tracker.goto_preset(),
                    asyncio.get_event_loop()
                )
            
            self.logger.info(f"OSC: Goto preset {preset_number or 'default'} command received")
        except Exception as e:
            self.logger.error(f"OSC goto preset error: {e}")
    
    def _handle_pan(self, unused_addr, *args):
        """Handle pan command"""
        try:
            if not args:
                self.logger.warning("OSC pan command requires angle argument")
                return
            
            angle = float(args[0])
            asyncio.run_coroutine_threadsafe(
                self.tracker.ptz_controller.pan_to(angle),
                asyncio.get_event_loop()
            )
            
            self.logger.info(f"OSC: Pan to {angle} degrees command received")
        except Exception as e:
            self.logger.error(f"OSC pan error: {e}")
    
    def _handle_status(self, unused_addr, *args):
        """Handle status request"""
        try:
            status = self.tracker.get_status()
            self.logger.info(f"OSC: Status request - {status}")
            
            # Optionally send status back to client
            # This would require an OSC client to send responses
            
        except Exception as e:
            self.logger.error(f"OSC status error: {e}")
    
    def _default_handler(self, unused_addr, *args):
        """Handle unrecognized OSC messages"""
        self.logger.warning(f"OSC: Unrecognized message: {unused_addr} {args}")
    
    async def start(self):
        """Start the OSC server"""
        if self.is_running:
            self.logger.warning("OSC server already running")
            return
        
        try:
            self.logger.info(f"Starting OSC server on {self.config['host']}:{self.config['port']}")
            
            self.server = ThreadingOSCUDPServer(
                (self.config['host'], self.config['port']),
                self.dispatcher
            )
            
            def run_server():
                try:
                    self.server.serve_forever()
                except Exception as e:
                    self.logger.error(f"OSC server error: {e}")
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            self.is_running = True
            
            # Give server time to start
            await asyncio.sleep(0.5)
            self.logger.info("OSC server started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start OSC server: {e}")
            raise
    
    async def stop(self):
        """Stop the OSC server"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping OSC server...")
        self.is_running = False
        
        if self.server:
            self.server.shutdown()
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)
        
        self.logger.info("OSC server stopped")
    
    def send_status_update(self, status: dict):
        """Send status update to OSC clients (if configured)"""
        # This would require setting up an OSC client to send messages
        # For now, just log the status
        if self.config.get('send_status_updates', False):
            self.logger.debug(f"Status update: {status}")