import sys
import time
import os
import threading
from race_engineer.telemetry_client import TORCSClient, MockTORCSClient
from race_engineer.engineer_agent import RaceEngineerAgent


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_telemetry(telemetry, frame, agent, conversation_history):
    """Print telemetry AND conversation history"""
    print(f"{'=' * 60}")
    print(f"  TORCS RACE ENGINEER - Frame {frame}")
    print(f"{'=' * 60}")
    print(f"  Speed:      {telemetry.get('speedX', 0):>7.2f} km/h")
    print(f"  RPM:        {telemetry.get('rpm', 0):>7.0f}")
    print(f"  Gear:       {telemetry.get('gear', 0):>7.0f}")
    print(f"  Track Pos:  {telemetry.get('trackPos', 0):>7.3f}")
    print(f"  Lap Time:   {telemetry.get('curLapTime', 0):>7.2f}s")
    print(f"  Last Lap:   {telemetry.get('lastLapTime', 0):>7.2f}s")
    print(f"  Fuel:       {telemetry.get('fuel', 0):>7.1f}L")
    print(f"  Damage:     {telemetry.get('damage', 0):>7.0f}")
    print(f"{'=' * 60}")

    # Statistics
    stats = agent.get_statistics()
    if stats['laps_completed'] > 0:
        print(f"\n  📊 Laps: {stats['laps_completed']} | "
              f"Best: {stats['best_lap']:.2f}s | "
              f"Avg: {stats['average_lap']:.2f}s")

    # Conversation history (show last 3 exchanges)
    print(f"\n{'=' * 60}")
    print(f"  🎙️  RADIO CONVERSATION")
    print(f"{'=' * 60}")

    if agent.is_generating:
        print("  🤔 Engineer is thinking...")
    
    # Show last 3 messages
    if conversation_history:
        display_history = conversation_history[-3:]  # Last 3 only
        for msg in display_history:
            timestamp = msg['timestamp']
            seconds_ago = int(time.time() - timestamp)
            
            if msg['type'] == 'question':
                print(f"\n  💬 YOU ({seconds_ago}s ago): {msg['text']}")
            else:  # response or auto-advice
                prefix = "🎙️  ENGINEER" if msg['type'] == 'response' else "📢 ENGINEER"
                print(f"  {prefix} ({seconds_ago}s ago): {msg['text']}")
    else:
        print("  Waiting for data...")
    
    print(f"{'=' * 60}")


class InputListener:
    """Listens for user input in a separate terminal window"""
    
    def __init__(self):
        self.question_queue = []
        self.running = True
        self.lock = threading.Lock()
        self.prompt_shown = False
        
    def listen(self):
        """Run in background thread to capture user input"""
        # Print instructions once
        print("\n" + "=" * 60)
        print("  INPUT WINDOW - Type questions here")
        print("=" * 60)
        print("  Examples: 'What's my damage?', 'Should I pit?'")
        print("  Type 'quit' to exit")
        print("=" * 60 + "\n")
        
        while self.running:
            try:
                user_input = input("Your question > ")
                
                if user_input.strip().lower() == 'quit':
                    self.running = False
                    break
                
                if user_input.strip():
                    with self.lock:
                        self.question_queue.append(user_input.strip())
                        print(f"  ✓ Question sent!\n")
                        
            except EOFError:
                break
            except Exception as e:
                print(f"\n[Input error: {e}]")
    
    def get_question(self):
        """Get next question from queue"""
        with self.lock:
            if self.question_queue:
                return self.question_queue.pop(0)
        return None
    
    def stop(self):
        self.running = False


def main():
    print("=" * 60)
    print("  TORCS RACE ENGINEER with Granite AI")
    print("  (Interactive Mode)")
    print("=" * 60)

    print("\nSelect mode:")
    print("1. Real TORCS")
    print("2. Mock mode")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        client = TORCSClient()
        print("\n⚠ Make sure TORCS is running!")
        input("Press Enter when ready...")
    else:
        client = MockTORCSClient()

    if not client.connect():
        print("\n✗ Failed to connect!")
        return

    # Initialize agent
    print("\n" + "=" * 60)
    agent = RaceEngineerAgent()

    print("=" * 60)
    print("\nIMPORTANT:")
    print("- Race telemetry will display in THIS window")
    print("- You can type questions HERE without clearing")
    print("- Last 3 radio messages will always be visible")
    input("\nPress Enter to start...\n")

    # Start input listener in background thread
    listener = InputListener()
    input_thread = threading.Thread(target=listener.listen, daemon=True)
    input_thread.start()

    frame = 0
    conversation_history = []  # Store all Q&A
    last_display_time = 0
    current_telemetry = None

    DISPLAY_REFRESH_RATE = 5.0

    try:
        while listener.running:
            telemetry = client.receive_telemetry()

            if telemetry:
                current_telemetry = telemetry
                
                # Check for user questions
                user_question = listener.get_question()
                if user_question:
                    # Add question to history
                    conversation_history.append({
                        'type': 'question',
                        'text': user_question,
                        'timestamp': time.time()
                    })
                    
                    # Get response from agent
                    response = agent.answer_question(user_question, telemetry)
                    
                    # Add response to history
                    conversation_history.append({
                        'type': 'response',
                        'text': response,
                        'timestamp': time.time()
                    })

                # Process automatic telemetry advice
                advice = agent.process_telemetry(telemetry)
                if advice:
                    conversation_history.append({
                        'type': 'auto_advice',
                        'text': advice,
                        'timestamp': time.time()
                    })

                # Refresh display at readable rate
                current_time = time.time()
                if current_time - last_display_time >= DISPLAY_REFRESH_RATE:
                    clear_screen()
                    print_telemetry(telemetry, frame, agent, conversation_history)
                    last_display_time = current_time

                # Send controls if real TORCS
                if isinstance(client, TORCSClient):
                    client.send_control(steer=0, accel=0.5, brake=0)

                frame += 1

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\nSession ended!")
    
    listener.stop()
    
    # Show full conversation history at end
    print("\n" + "=" * 60)
    print("  FULL CONVERSATION HISTORY")
    print("=" * 60)
    for msg in conversation_history:
        if msg['type'] == 'question':
            print(f"\n💬 YOU: {msg['text']}")
        else:
            print(f"🎙️  ENGINEER: {msg['text']}")
    
    stats = agent.get_statistics()
    if stats['laps_completed'] > 0:
        print(f"\n📊 Final Stats:")
        print(f"   Laps:    {stats['laps_completed']}")
        print(f"   Best:    {stats['best_lap']:.2f}s")
        print(f"   Average: {stats['average_lap']:.2f}s")

    client.close()
    print("\n✓ Done!")

if __name__ == "__main__":
    main()