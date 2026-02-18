import socket
import sys
import os

class TORCSClient:
    def __init__(self, host='localhost', port=3001):
        """
        Windows-compatible TORCS client
        
        Args:
            host: TORCS server host
            port: TORCS server port (default 3001)
        """
        self.host = host
        self.port = port
        self.sock = None
        
    def connect(self):
        """Connect to TORCS server"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(1.0)
            
            # Send initialization string
            init_angles = [-90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 
                          5, 10, 15, 20, 30, 45, 60, 75, 90]
            init_string = f"SCR(init {' '.join(map(str, init_angles))})"
            self.sock.sendto(init_string.encode('utf-8'), (self.host, self.port))
            
            print(f"✓ Connected to TORCS at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def receive_telemetry(self):
        """Receive and parse telemetry from TORCS"""
        if not self.sock:
            print("Not connected! Call connect() first.")
            return None
            
        try:
            data, addr = self.sock.recvfrom(4096)
            return self.parse_server_string(data.decode('utf-8'))
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None
    
    def parse_server_string(self, server_string):
        """Parse SCR protocol string into dictionary"""
        telemetry = {}
        server_string = server_string.strip('()')
        items = server_string.split(')(')
        
        for item in items:
            parts = item.split(' ', 1)
            if len(parts) == 2:
                key = parts[0]
                value_str = parts[1]
                
                if value_str.startswith('('):
                    # Array value
                    values = value_str.strip('()').split()
                    try:
                        telemetry[key] = [float(v) for v in values]
                    except ValueError:
                        telemetry[key] = values
                else:
                    # Single value
                    try:
                        telemetry[key] = float(value_str)
                    except ValueError:
                        telemetry[key] = value_str
        
        return telemetry
    
    def send_control(self, steer=0, accel=0, brake=0, gear=1, clutch=0):
        """Send control commands to TORCS"""
        if not self.sock:
            return
            
        control_string = (
            f"(accel {accel})(brake {brake})(gear {gear})"
            f"(steer {steer})(clutch {clutch})(meta 0)"
        )
        try:
            self.sock.sendto(control_string.encode('utf-8'), (self.host, self.port))
        except Exception as e:
            print(f"Error sending control: {e}")
    
    def close(self):
        """Close the connection"""
        if self.sock:
            self.sock.close()
            print("Connection closed")


# Mock client for testing without TORCS
class MockTORCSClient:
    """Simulated TORCS client for testing agent logic without TORCS"""
    
    def __init__(self):
        self.frame = 0
        self.lap_time = 0
        
    def connect(self):
        print("✓ Mock client initialized (no TORCS needed)")
        return True
    
    def receive_telemetry(self):
        """Generate fake but realistic telemetry"""
        import random
        import math
        
        self.frame += 1
        self.lap_time += 0.02  # 50 Hz updates
        
        # Simulate a lap
        progress = (self.lap_time % 90) / 90  # 90 second lap
        
        telemetry = {
            'speedX': 150 + 50 * math.sin(progress * 2 * math.pi),
            'speedY': random.uniform(-2, 2),
            'speedZ': random.uniform(-1, 1),
            'rpm': 6000 + 2000 * math.sin(progress * 4 * math.pi),
            'gear': int(3 + 2 * math.sin(progress * 2 * math.pi)),
            'trackPos': random.uniform(-0.5, 0.5),
            'angle': random.uniform(-0.1, 0.1),
            'damage': 0,
            'distRaced': progress * 5000,  # 5km track
            'fuel': 100 - (self.lap_time / 10),
            'lastLapTime': 89.5 + random.uniform(-2, 2),
            'curLapTime': self.lap_time % 90,
            'racePos': 3,
        }
        
        return telemetry
    
    def send_control(self, steer=0, accel=0, brake=0, gear=1, clutch=0):
        pass  # Mock does nothing
    
    def close(self):
        print("Mock client closed")