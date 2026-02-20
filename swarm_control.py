import subprocess, time, os, shutil, sys

# ---------------------------------------------------------------------------
NUM_CARS   = 1       # Number of cars. Must match SCR server drivers in TORCS.
START_PORT = 3001
TORCS_ROOT = r"C:\Users\User\Downloads\torcs\torcs"
TORCS_EXE  = os.path.join(TORCS_ROOT, "wtorcs.exe")
BASE_DIR   = r"C:\Users\User\Downloads\torcs\gym_torcs"
PYTHON     = r"C:/Users/User/AppData/Local/Microsoft/WindowsApps/python3.11.exe"

# ---------------------------------------------------------------------------

def kill_agents(agent_procs):
    print("\nStopping agents...")
    for proc in agent_procs:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except:
            try: proc.kill()
            except: pass
    print(f"  {len(agent_procs)} agents stopped.")


def find_and_sync_champion():
    print("\nAnalysing swarm performance...")
    best_time   = float('inf')
    winner_port = None

    for fname in os.listdir(BASE_DIR):
        if fname.startswith("laptime_") and fname.endswith(".txt"):
            try:
                port  = fname.split("_")[1].split(".")[0]
                fpath = os.path.join(BASE_DIR, fname)
                with open(fpath) as f:
                    lap_time = float(f.read().strip())
                print(f"  Port {port}: {lap_time:.2f}s")
                if lap_time < best_time:
                    best_time   = lap_time
                    winner_port = port
            except:
                continue

    if winner_port:
        print(f"\nCHAMPION: Port {winner_port} — lap time {best_time:.2f}s")
        winner_brain = os.path.join(BASE_DIR, f"brain_{winner_port}_evolved.pth")
        master_path  = os.path.join(BASE_DIR, "master_brain.pth")
        if os.path.exists(winner_brain):
            shutil.copy(winner_brain, master_path)
            print("Champion brain promoted to master_brain.pth")
        else:
            print("WARNING: Winner brain file missing — master_brain.pth unchanged.")
    else:
        print("No lap times recorded — master_brain.pth unchanged.")

    # Clean up for next generation
    cleaned = 0
    for fname in os.listdir(BASE_DIR):
        if ("_evolved.pth" in fname or
                fname.startswith("laptime_") or
                fname.startswith("ready_")):
            try:
                os.remove(os.path.join(BASE_DIR, fname))
                cleaned += 1
            except:
                pass
    print(f"Cleaned {cleaned} files for next generation.")


def launch():
    agent_procs = []

    # --- Kill any leftover processes from last time ---
    os.system("taskkill /F /IM wtorcs.exe /T >nul 2>&1")
    time.sleep(1)

    # --- Check results from last session ---
    find_and_sync_champion()

    # --- Launch TORCS ---
    print(f"\nLaunching TORCS...")
    torcs_proc = subprocess.Popen([TORCS_EXE, "-p", str(START_PORT)], cwd=TORCS_ROOT)

    print(f"""
TORCS SETUP:
  1. Race -> Quick Race -> Configure
  2. Set exactly {NUM_CARS} SCR Server driver(s) in the selected list
  3. Pick your track
  4. Click New Race / Start
  5. Wait for the loading screen to appear, then come back here
""")
    input("--- Press ENTER when the TORCS loading screen is visible ---\n")

    # --- Launch agents one at a time, waiting for each to confirm connection ---
    agent_script = os.path.join(BASE_DIR, "torcs_hybrid_learner.py")

    for i in range(NUM_CARS):
        port       = START_PORT + i
        ready_path = os.path.join(BASE_DIR, f"ready_{port}.flag")

        # Clear any stale flag from a previous run
        if os.path.exists(ready_path):
            os.remove(ready_path)

        print(f"Launching agent {i+1} of {NUM_CARS} on port {port}...")
        proc = subprocess.Popen(
            [PYTHON, agent_script, str(port)],
            cwd=BASE_DIR,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        agent_procs.append(proc)

        # Wait for this agent to write its ready flag before launching the next.
        # The agent writes the flag the moment it receives 'identified' from TORCS.
        print(f"  Waiting for agent {port} to connect", end="", flush=True)
        start = time.time()
        while not os.path.exists(ready_path):
            time.sleep(0.5)
            print(".", end="", flush=True)
            if time.time() - start > 90:
                print(" TIMED OUT")
                print("  Check the agent window for errors.")
                break
        else:
            print(" CONNECTED!")

        # Brief pause before launching the next agent
        if i < NUM_CARS - 1:
            time.sleep(1.0)

    print(f"\n{'='*50}")
    print(f"All {NUM_CARS} agent(s) connected and racing!")
    print(f"Watch the agent console window(s) for lap times.")
    print(f"{'='*50}")
    print()
    input(">>> When you are ready to STOP, press ENTER here <<<\n")

    # --- Kill agents BEFORE reading results to avoid race conditions ---
    kill_agents(agent_procs)
    time.sleep(1.5)

    # --- Save the best brain ---
    find_and_sync_champion()

    # --- Stop TORCS ---
    try:
        torcs_proc.terminate()
    except:
        os.system("taskkill /F /IM wtorcs.exe /T >nul 2>&1")

    print("\nDone! Run this script again to start the next generation.")


if __name__ == "__main__":
    launch()