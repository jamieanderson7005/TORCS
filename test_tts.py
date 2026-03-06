import pyttsx3

print("Testing pyttsx3...")

engine = pyttsx3.init()
print("✓ Engine initialized")

# List available voices
voices = engine.getProperty('voices')
print(f"\nAvailable voices: {len(voices)}")
for i, voice in enumerate(voices):
    print(f"{i}: {voice.name}")

# Try to speak
print("\nAttempting to speak...")
engine.say("Copy that, pushing hard")
engine.runAndWait()
print("✓ Done")

