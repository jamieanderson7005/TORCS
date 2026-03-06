import pyttsx3
import threading

class RaceEngineerTTS:
    def __init__(self):
        print("[TTS] Initializing engine...")
        self.engine = pyttsx3.init()
        
        # Configure voice
        voices = self.engine.getProperty('voices')
        print(f"[TTS] Found {len(voices)} voices")
        
        for voice in voices:
            if 'male' in voice.name.lower() and 'female' not in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                print(f"[TTS] Selected voice: {voice.name}")
                break
        
        self.engine.setProperty('rate', 175)
        self.engine.setProperty('volume', 0.9)
        
        self.is_speaking = False
        print("[TTS] Engine ready!")
        
    def speak(self, text):
        """Speak text in a separate thread so it doesn't block"""
        print(f"[TTS] Speaking: {text}")  # DEBUG
        
        def _speak():
            self.is_speaking = True
            print("[TTS] Thread started")  # DEBUG
            try:
                self.engine.say(text)
                self.engine.runAndWait()
                print("[TTS] Speech completed")  # DEBUG
            except Exception as e:
                print(f"[TTS ERROR] {e}")
            finally:
                self.is_speaking = False
        
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    
    def speak_blocking(self, text):
        """Speak and wait for it to finish"""
        print(f"[TTS BLOCKING] Speaking: {text}")
        self.is_speaking = True
        self.engine.say(text)
        self.engine.runAndWait()
        self.is_speaking = False
        print("[TTS BLOCKING] Done")