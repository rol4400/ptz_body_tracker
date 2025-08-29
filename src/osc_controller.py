#!/usr/bin/env python3
"""
OSC Controller for PTZ Camera Tracking System
Provides OSC interface for remote control
"""

import asyncio
import logging
import socket
import threading
import queue
from typing import Optional, Callable

try:
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import BlockingOSCUDPServer
    from pythonosc import udp_client
    OSC_AVAILABLE = True
except ImportError:
    # Fallback if pythonosc is not available
    OSC_AVAILABLE = False
    print("Warning: pythonosc not available, OSC functionality disabled")


class OSCController:
    """OSC controller for PTZ tracking system"""
    
    def __init__(self, config: dict, tracking_system=None):
        self.config = config
        self.tracking_system = tracking_system
        self.logger = logging.getLogger(__name__)
        
        # Command queue for thread-safe communication
        self.command_queue = queue.Queue()
        
        if not OSC_AVAILABLE:
            self.logger.warning("OSC functionality disabled - pythonosc not available")
            self.enabled = False
            return
        
        self.enabled = True
        
        # OSC server configuration
        self.host = config.get('osc', {}).get('host', '127.0.0.1')
        self.port = config.get('osc', {}).get('port', 8081)
        
        # OSC client configuration for sending status
        self.client_host = config.get('osc', {}).get('client_host', '127.0.0.1')
        self.client_port = config.get('osc', {}).get('client_port', 8082)
        
        self.server: Optional[BlockingOSCUDPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.running = False
        
    def setup_dispatcher(self) -> Dispatcher:
        """Setup OSC message dispatcher"""
        dispatcher = Dispatcher()
        
        # Control commands
        dispatcher.map("/ptz/start", self.handle_start)
        dispatcher.map("/ptz/stop", self.handle_stop)
        dispatcher.map("/ptz/relock", self.handle_relock)
        dispatcher.map("/ptz/status", self.handle_status_request)
        dispatcher.map("/ptz/preset", self.handle_preset)
        
        # Query commands
        dispatcher.map("/ptz/people_count", self.handle_people_count_request)
        dispatcher.map("/ptz/lock_status", self.handle_lock_status_request)
        
        return dispatcher
    
    def start(self):
        """Start OSC server"""
        if not self.enabled:
            self.logger.info("OSC server disabled")
            return False
            
        try:
            dispatcher = self.setup_dispatcher()
            self.server = BlockingOSCUDPServer((self.host, self.port), dispatcher)
            
            self.running = True
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()
            
            self.logger.info(f"OSC server started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start OSC server: {e}")
            return False
    
    def _server_loop(self):
        """OSC server loop"""
        try:
            while self.running and self.server:
                self.server.serve_forever()
        except Exception as e:
            self.logger.error(f"OSC server error: {e}")
    
    def stop(self):
        """Stop OSC server"""
        if not self.enabled:
            return
            
        self.running = False
        if self.server:
            self.server.shutdown()
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        self.logger.info("OSC server stopped")
    
    # OSC Message Handlers
    
    def handle_start(self, unused_addr, *args):
        """Handle start tracking command"""
        if self.tracking_system:
            self.command_queue.put(('start', None))
            self.logger.info("OSC: Start tracking command received")
    
    def handle_stop(self, unused_addr, *args):
        """Handle stop tracking command"""
        if self.tracking_system:
            self.command_queue.put(('stop', None))
            self.logger.info("OSC: Stop tracking command received")
    
    def handle_relock(self, unused_addr, *args):
        """Handle relock command"""
        if self.tracking_system:
            self.command_queue.put(('relock', None))
            self.logger.info("OSC: Relock command received")
    
    def handle_preset(self, unused_addr, *args):
        """Handle preset recall command"""
        if self.tracking_system:
            preset_number = args[0] if args else 4  # Default to preset 4
            self.command_queue.put(('preset', preset_number))
            self.logger.info(f"OSC: Preset {preset_number} command received")
    
    def handle_status_request(self, unused_addr, *args):
        """Handle status request"""
        self.send_status_update()
    
    def handle_people_count_request(self, unused_addr, *args):
        """Handle people count request"""
        if self.tracking_system:
            count = self.tracking_system.people_count
            self.send_osc_message("/ptz/people_count", count)
    
    def handle_lock_status_request(self, unused_addr, *args):
        """Handle lock status request"""
        if self.tracking_system:
            status = self.tracking_system.lock_status
            self.send_osc_message("/ptz/lock_status", status)
    
    def send_status_update(self):
        """Send complete status update"""
        if self.tracking_system:
            status = self.tracking_system.get_status()
            
            # Send individual status messages
            self.send_osc_message("/ptz/tracking", 1 if status["is_tracking"] else 0)
            self.send_osc_message("/ptz/people_count", status["people_count"])
            self.send_osc_message("/ptz/lock_status", 1 if status["primary_person_id"] else 0)
            self.send_osc_message("/ptz/camera_moving", 1 if status["camera_moving"] else 0)
            
            # Send combined status
            self.send_osc_message("/ptz/status", 
                                status["is_tracking"], 
                                status["people_count"], 
                                1 if status["primary_person_id"] else 0)
    
    def send_osc_message(self, address: str, *args):
        """Send OSC message to client"""
        try:
            import socket
            
            client = udp_client.SimpleUDPClient(self.client_host, self.client_port)
            client.send_message(address, args if len(args) > 1 else args[0] if args else None)
            
        except Exception as e:
            self.logger.debug(f"Failed to send OSC message: {e}")
    
    async def process_command_queue(self):
        """Process commands from the OSC queue in the main async loop"""
        while True:
            try:
                # Non-blocking check for commands
                command, args = self.command_queue.get_nowait()
                
                if command == 'start':
                    await self.tracking_system.start_tracking()
                    self.send_status_update()
                elif command == 'stop':
                    await self.tracking_system.stop_tracking()
                    self.send_status_update()
                elif command == 'relock':
                    await self.tracking_system.lock_primary_person()
                    self.send_status_update()
                elif command == 'preset':
                    await self.tracking_system.goto_preset(args)
                    self.send_status_update()
                    
            except queue.Empty:
                # No commands to process
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error processing OSC command: {e}")
                await asyncio.sleep(0.1)