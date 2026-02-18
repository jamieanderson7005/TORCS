import sys
import time
import os
from race_engineer.telemetry_client import TORCSClient, MockTORCSClient
from race_engineer.engineer_agent import RaceEngineerAgent

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_telemetry(telemetry, frame, agent, last_advice, last_advice_time):
    """Print telemetry AND advice together every frame"""
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

    # Radio section
    print(f"\n{'=' * 60}")
    print(f"    ENGINEER RADIO")
    print(f"{'=' * 60}")

    if agent.is_generating:
        print("   Engineer is thinking...")
    elif last_advice:
        # Show how long ago advice was given
        seconds_ago = int(time.time() - last_advice_time)
        print(f"  {last_advice}")
        print(f"\n  (Received {seconds_ago}s ago)")
    else:
        print("  Waiting for data...")

    print(f"{'=' * 60}")

def main():
    print("=" * 60)
    print("  TORCS RACE ENGINEER with Granite AI")
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
    input("\nPress Enter to start...\n")

    frame = 0
    last_advice = None
    last_advice_time = 0
    last_display_time = 0

    # How often to refresh the screen (in seconds)
    DISPLAY_REFRESH_RATE = 2.0  # Refresh every 1 second - easy to read

    try:
        while True:
            telemetry = client.receive_telemetry()

            if telemetry:
                # Always process telemetry at full speed
                advice = agent.process_telemetry(telemetry)

                # Update advice if new one arrived
                if advice:
                    last_advice = advice
                    last_advice_time = time.time()

                # Only refresh screen at a readable rate
                current_time = time.time()
                if current_time - last_display_time >= DISPLAY_REFRESH_RATE:
                    clear_screen()
                    print_telemetry(
                        telemetry, frame, agent,
                        last_advice, last_advice_time
                    )
                    last_display_time = current_time

                # Send controls if real TORCS
                if isinstance(client, TORCSClient):
                    client.send_control(steer=0, accel=0.5, brake=0)

                frame += 1

            time.sleep(0.02)  # Keep telemetry processing at 50Hz

    except KeyboardInterrupt:
        print("\n\nSession ended!")
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