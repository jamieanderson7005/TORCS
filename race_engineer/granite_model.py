import ollama

class GraniteRaceEngineer:
    def __init__(self, model_name="granite3.3:2b"):
        self.model_name = model_name
        print(f"Connecting to Ollama with {model_name}...")
        
        # Test connection
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt="Say 'Ready' and nothing else.",
                options={"num_predict": 5}
            )
            print(f"✓ Model ready!")
        except Exception as e:
            print(f"✗ Ollama error: {e}")
            print("Make sure Ollama is running and the model is pulled!")
            raise
    
    def analyze_telemetry(self, telemetry, context=""):
        prompt = self._build_telemetry_prompt(telemetry, context)
        return self.generate(prompt, max_tokens=60)
    
    def _build_telemetry_prompt(self, telemetry, context):
        return f"""You are an F1 race engineer giving radio advice. Be brief and direct.

Telemetry:
- Speed: {telemetry.get('speedX', 0):.1f} km/h
- RPM: {telemetry.get('rpm', 0):.0f}
- Gear: {telemetry.get('gear', 0)}
- Track Position: {telemetry.get('trackPos', 0):.3f}
- Lap Time: {telemetry.get('curLapTime', 0):.2f}s
- Fuel: {telemetry.get('fuel', 0):.1f}L
- Damage: {telemetry.get('damage', 0):.0f}

Situation: {context}

Radio message (1-2 sentences max):"""
    
    def generate(self, prompt, max_tokens=60):
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["\n\n"]  # Stop at double newline
                }
            )
            return response['response'].strip()
        except Exception as e:
            return f"[Engineer unavailable: {e}]"
    
    def compare_laps(self, current_lap_time, best_lap_time):
        delta = current_lap_time - best_lap_time
        prompt = f"""F1 radio message for lap completed in {current_lap_time:.2f}s.
Best lap: {best_lap_time:.2f}s. Delta: {delta:+.2f}s.
One sentence response:"""
        return self.generate(prompt, max_tokens=40)