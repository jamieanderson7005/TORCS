class TelemetryAnalyzer:
    """Tools for analyzing racing telemetry"""
    
    def __init__(self):
        self.lap_times = []
        self.sector_times = []
        self.best_lap = float('inf')
        
    def is_off_track(self, telemetry):
        """Check if car is off track"""
        track_pos = telemetry.get('trackPos', 0)
        return abs(track_pos) > 1.0
    
    def is_going_slow(self, telemetry, threshold=50):
        """Check if car is going unusually slow"""
        speed = telemetry.get('speedX', 0)
        return speed < threshold
    
    def is_high_damage(self, telemetry, threshold=5000):
        """Check if car has significant damage"""
        damage = telemetry.get('damage', 0)
        return damage > threshold
    
    def is_low_fuel(self, telemetry, threshold=10):
        """Check if fuel is running low"""
        fuel = telemetry.get('fuel', 0)
        return fuel < threshold
    
    def analyze_sector(self, telemetry):
        """Determine which sector the car is in based on lap progress"""
        lap_time = telemetry.get('curLapTime', 0)
        
        # Estimate sector based on lap time (assuming ~90s lap)
        if lap_time < 30:
            return 1
        elif lap_time < 60:
            return 2
        else:
            return 3
    
    def record_lap(self, lap_time):
        """Record a lap time"""
        self.lap_times.append(lap_time)
        if lap_time < self.best_lap:
            self.best_lap = lap_time
            return True  # New personal best!
        return False
    
    def get_lap_delta(self, current_time):
        """Get delta to best lap"""
        if self.best_lap == float('inf'):
            return 0
        return current_time - self.best_lap
    
    def calculate_fuel_laps_remaining(self, telemetry, fuel_per_lap=5.0):
        """Estimate how many laps can be completed with current fuel"""
        fuel = telemetry.get('fuel', 0)
        return fuel / fuel_per_lap
    
    def get_statistics(self):
        """Get racing statistics"""
        if not self.lap_times:
            return {
                'laps_completed': 0,
                'best_lap': None,
                'average_lap': None
            }
        
        return {
            'laps_completed': len(self.lap_times),
            'best_lap': min(self.lap_times),
            'average_lap': sum(self.lap_times) / len(self.lap_times),
            'last_lap': self.lap_times[-1] if self.lap_times else None
        }