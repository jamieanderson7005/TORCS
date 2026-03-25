import sys
import time
import os
import threading
from race_engineer.telemetry_client import TORCSClient, MockTORCSClient
from race_engineer.engineer_agent import RaceEngineerAgent

def championship_driver(telemetry):
    """
    Better autopilot based on SCR championship algorithms
    Returns: (steer, accel, brake, gear)
    """
    # Get telemetry with type conversion
    angle = float(telemetry.get('angle', 0))
    track_pos = float(telemetry.get('trackPos', 0))
    speed_x = float(telemetry.get('speedX', 0))
    speed_y = float(telemetry.get('speedY', 0))
    speed_z = float(telemetry.get('speedZ', 0))
    rpm = float(telemetry.get('rpm', 0))
    current_gear = int(telemetry.get('gear', 1))
    
    # Get track sensors
    track_sensors = telemetry.get('track', [])
    if track_sensors:
        try:
            track_sensors = [float(x) for x in track_sensors]
        except (ValueError, TypeError):
            track_sensors = [100] * 19  # Default safe values
    else:
        track_sensors = [100] * 19
    
    # Calculate speed magnitude
    speed = (speed_x**2 + speed_y**2 + speed_z**2)**0.5
    
    # === STEERING ===
    # Use track edge sensors for smarter steering
    if len(track_sensors) >= 19:
        # Sensors indexed 0-18, where 9 is center
        rxSensor = track_sensors[10:]  # Right side sensors
        cSensor = track_sensors[9]     # Center sensor
        lxSensor = track_sensors[:9]   # Left side sensors
        
        # Find the track direction
        right_sum = sum(rxSensor) / len(rxSensor)
        left_sum = sum(lxSensor) / len(lxSensor)
        
        # Target angle based on track direction
        target_angle = (right_sum - left_sum) / 200.0
        
        # Steering calculation
        steer = (target_angle - angle) * 5.0
        
        # Add track position correction
        steer += -track_pos * 0.5
        
        # If very close to edge, aggressive correction
        if abs(track_pos) > 0.9:
            steer += -track_pos * 3.0
    else:
        # Fallback steering
        steer = -angle * 5 - track_pos * 2
    
    # Limit steering
    steer = max(-1, min(1, steer))
    
    # === SPEED CONTROL ===
    # Calculate track curvature from sensors
    if len(track_sensors) >= 19:
        # Use center sensors to detect turns
        front_dist = track_sensors[9]
        left_front = track_sensors[7]
        right_front = track_sensors[11]
        
        # Estimate turn radius
        if front_dist < 100:
            # Sharp turn ahead
            target_speed = 60 + front_dist * 0.5
        elif min(left_front, right_front) < 80:
            # Medium turn
            target_speed = 100
        else:
            # Straight or gentle turn
            target_speed = 200
        
        # Adjust for current angle (already in turn)
        if abs(angle) > 0.5:
            target_speed = min(target_speed, 70)
        elif abs(angle) > 0.3:
            target_speed = min(target_speed, 100)
    else:
        target_speed = 150
    
    # Adjust for track position
    if abs(track_pos) > 0.8:
        target_speed *= 0.7  # Slow down when off-center
    
    # Calculate acceleration/brake
    speed_diff = target_speed - speed
    
    if speed_diff > 30:
        accel = 1.0
        brake = 0
    elif speed_diff > 10:
        accel = 0.7
        brake = 0
    elif speed_diff > -10:
        accel = 0.4
        brake = 0
    elif speed_diff > -30:
        accel = 0
        brake = 0.3
    else:
        accel = 0
        brake = 0.7
    
    # Emergency brake
    if abs(track_pos) > 0.95:
        accel = 0
        brake = 1.0
    
    # === GEAR SHIFTING ===
    # More sophisticated shifting
    if current_gear < 1:
        gear = 1
    elif rpm > 9000:  # Redline
        gear = min(current_gear + 1, 6)
    elif rpm > 7500 and current_gear < 6:  # Optimal shift point
        gear = current_gear + 1
    elif rpm < 2000 and current_gear > 2:  # Downshift to maintain power
        gear = current_gear - 1
    elif rpm < 1500 and current_gear > 1:  # Emergency downshift
        gear = max(current_gear - 1, 1)
    else:
        gear = current_gear
    
    # Don't shift down at high speed
    if speed > 80 and gear < current_gear:
        gear = current_gear
    
    return steer, accel, brake, gear


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
                    #steer, accel, brake, gear = championship_driver(telemetry)                    
                    pass

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