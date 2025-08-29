#!/usr/bin/env python3
"""
Quick OSC test script to send commands to the PTZ system
"""

import time
from pythonosc import udp_client

def main():
    # Create OSC client
    client = udp_client.SimpleUDPClient("127.0.0.1", 8081)
    
    print("Sending OSC start command...")
    client.send_message("/ptz/start", None)
    
    print("Waiting 5 seconds...")
    time.sleep(5)
    
    print("Sending OSC status request...")
    client.send_message("/ptz/status", None)
    
    print("OSC commands sent!")

if __name__ == "__main__":
    main()
