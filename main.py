#!/usr/bin/env python3
"""
PTZ Camera Tracking System
Main entry point for the application
"""

import argparse
import sys
import logging
from pathlib import Path
import asyncio
import signal
import json

from src.tracker import PTZTracker
from src.api_server import APIServer
from src.osc_server import OSCServer


def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ptz_tracker.log')
        ]
    )


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)


async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='PTZ Camera Tracking System')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--start', action='store_true', help='Start tracking')
    parser.add_argument('--stop', action='store_true', help='Stop tracking')
    parser.add_argument('--lock', action='store_true', help='Lock onto primary person')
    parser.add_argument('--preset', action='store_true', help='Go to default preset')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon with API/OSC servers')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(args.debug or config.get('system', {}).get('debug', False))
    
    logger = logging.getLogger(__name__)
    logger.info("Starting PTZ Camera Tracking System")
    
    # Initialize tracker
    tracker = PTZTracker(config)
    
    try:
        await tracker.initialize()
        
        # Handle single commands
        if args.start:
            await tracker.start_tracking()
            logger.info("Tracking started")
            return
        elif args.stop:
            await tracker.stop_tracking()
            logger.info("Tracking stopped")
            return
        elif args.lock:
            await tracker.lock_primary_person()
            logger.info("Locked onto primary person")
            return
        elif args.preset:
            await tracker.goto_preset()
            logger.info("Moved to default preset")
            return
        
        # Run as daemon with servers
        if args.daemon:
            logger.info("Starting daemon mode with API and OSC servers")
            
            # Initialize servers
            api_server = APIServer(tracker, config['api'])
            osc_server = OSCServer(tracker, config['osc'])
            
            # Setup signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info("Received signal, shutting down...")
                asyncio.create_task(shutdown(tracker, api_server, osc_server))
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Start servers
            await asyncio.gather(
                api_server.start(),
                osc_server.start(),
                tracker.run()
            )
        else:
            # Default: start tracking
            await tracker.start_tracking()
            await tracker.run()
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        await tracker.cleanup()


async def shutdown(tracker, api_server, osc_server):
    """Graceful shutdown"""
    logger = logging.getLogger(__name__)
    logger.info("Shutting down...")
    
    await tracker.stop_tracking()
    await tracker.cleanup()
    await api_server.stop()
    await osc_server.stop()
    
    sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        sys.exit(0)