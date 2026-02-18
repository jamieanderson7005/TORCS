import time
import threading
from .granite_model import GraniteRaceEngineer
from .tools import TelemetryAnalyzer

class RaceEngineerAgent:
    def __init__(self, model_name="granite3.3:2b"):
        print("Initializing Race Engineer Agent...")
        
        self.engineer = GraniteRaceEngineer(model_name)
        self.analyzer = TelemetryAnalyzer()
        
        # Timing
        self.last_advice_time = 0
        self.advice_interval = 15  # Longer interval for CPU (15 seconds)
        self.last_lap_time = 0
        
        # Threading - so AI doesn't block telemetry loop
        self.is_generating = False
        self.latest_advice = None
        self.advice_lock = threading.Lock()
        
        print("✓ Race Engineer ready!\n")
    
    def should_speak(self, telemetry):
        """Decide if engineer should give advice"""
        # Don't speak if already generating
        if self.is_generating:
            return False, None
        
        current_time = time.time()
        
        # Critical situations
        if self.analyzer.is_off_track(telemetry):
            return True, "off_track"
        
        if self.analyzer.is_high_damage(telemetry):
            return True, "damage"
        
        if self.analyzer.is_low_fuel(telemetry):
            return True, "low_fuel"
        
        # Lap completion
        lap_time = telemetry.get('curLapTime', 0)
        if lap_time < self.last_lap_time and self.last_lap_time > 10:
            self.last_lap_time = lap_time
            return True, "lap_complete"
        self.last_lap_time = lap_time
        
        # Regular interval
        if current_time - self.last_advice_time > self.advice_interval:
            return True, "interval"
        
        return False, None
    
    def _generate_advice_thread(self, telemetry, reason):
        """Run in background thread so telemetry loop isn't blocked"""
        try:
            self.is_generating = True
            advice = self._build_advice(telemetry, reason)
            
            with self.advice_lock:
                self.latest_advice = advice
                
        except Exception as e:
            print(f"Error generating advice: {e}")
        finally:
            self.is_generating = False
    
    def _build_advice(self, telemetry, reason):
        """Build context and generate advice"""
        context = ""
        
        if reason == "off_track":
            context = "WARNING: Car is off track!"
        
        elif reason == "damage":
            damage = telemetry.get('damage', 0)
            context = f"Car has taken damage: {damage:.0f}"
        
        elif reason == "low_fuel":
            fuel = telemetry.get('fuel', 0)
            laps_left = self.analyzer.calculate_fuel_laps_remaining(telemetry)
            context = f"Low fuel: {fuel:.1f}L left (~{laps_left:.1f} laps)"
        
        elif reason == "lap_complete":
            last_lap = telemetry.get('lastLapTime', 0)
            is_pb = self.analyzer.record_lap(last_lap)
            delta = self.analyzer.get_lap_delta(last_lap)
            
            if is_pb:
                context = f"Personal best! {last_lap:.2f}s"
            else:
                context = f"Lap complete: {last_lap:.2f}s (delta: {delta:+.2f}s)"
        
        elif reason == "interval":
            sector = self.analyzer.analyze_sector(telemetry)
            context = f"Sector {sector} update."
        
        return self.engineer.analyze_telemetry(telemetry, context)
    
    def process_telemetry(self, telemetry):
        """
        Process telemetry - returns advice if ready, triggers generation if needed
        """
        should_speak, reason = self.should_speak(telemetry)
        
        # Trigger background generation
        if should_speak:
            self.last_advice_time = time.time()
            thread = threading.Thread(
                target=self._generate_advice_thread,
                args=(telemetry, reason),
                daemon=True
            )
            thread.start()
        
        # Return any advice that's ready
        with self.advice_lock:
            if self.latest_advice:
                advice = self.latest_advice
                self.latest_advice = None  # Clear after returning
                return advice
        
        return None
    
    def get_statistics(self):
        return self.analyzer.get_statistics()