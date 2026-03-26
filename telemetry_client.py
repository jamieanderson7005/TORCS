class FileTORCSClient:
    """
    Reads telemetry from torcs_telemetry.json written by torcs_continuous.py.
    Use this when the driver script already owns port 3001.
    Both scripts must be in the same folder (or pass telemetry_path manually).
    """
    STALE_THRESHOLD = 3.0  # seconds before data is considered stale
    def __init__(self, telemetry_path: str = None):
        self.telemetry_path = telemetry_path or r"C:\TORCS-main\torcs_telemetry.json"
    def connect(self) -> bool:
        print(f"✓ File client — reading from:\n  {self.telemetry_path}")
        print("  Make sure torcs_continuous.py --drive is running in another window.")
        return True

    def receive_telemetry(self) -> dict | None:
        import json as _json, time as _time
        try:
            with open(self.telemetry_path) as f:
                parsed = _json.load(f)
            if _time.time() - parsed.get("timestamp", 0) > self.STALE_THRESHOLD:
                return None  # Driver has paused or stopped
            return parsed
        except (FileNotFoundError, ValueError):
            return None
        except Exception as e:
            print(f"[FileClient] read error: {e}")
            return None
