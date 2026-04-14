import time
import threading
import re
from telemetry_client import FileTORCSClient


class LiveCommentator:

    def __init__(self, granite_model, cooldown=1.0, telemetry_client=None):
        self.granite_model = granite_model
        self.cooldown = cooldown

        self.last_corner_time = 0
        self.in_corner = False
        self.latest_comment = None

        # Smooth corner detection memory
        self.prev_angle = 0
        self.angle_trend = 0

        self.telemetry_client = telemetry_client or FileTORCSClient()
        self.telemetry_client.connect()

    # ==========================
    # READ TELEMETRY
    # ==========================
    def read_telemetry(self):
        if self.telemetry_client:
            return self.telemetry_client.receive_telemetry()
        return None

    # ==========================
    # CORNER DETECTION
    # ==========================
    def detect_corner(self, S):

        track = S.get("track", [200] * 19)
        angle = S.get("angle", 0)

        # Ignore broken sensor data
        if min(track) < 0:
            return False

        # --- Sensor-based detection (sharp corners) ---
        left = min(track[0:5])
        right = min(track[14:19])
        curvature = abs(right - left)

        fwd_min = min(track[7:12])

        left_diag = min(track[3:6])
        right_diag = min(track[13:16])
        diag_asymmetry = abs(left_diag - right_diag)

        # --- NEW: smooth corner detection ---
        angle_delta = abs(angle - self.prev_angle)
        self.angle_trend = 0.8 * self.angle_trend + angle_delta
        self.prev_angle = angle

        speed = S.get("speedX", 0)

        # --- FINAL DECISION ---
        if (
            curvature > 6 or                  # sharp corner
            fwd_min < 70 or                  # tight ahead
            diag_asymmetry > 10 or           # early bend
            abs(angle) > 0.04 or             # already turning
            self.angle_trend > 0.01 or       # ✅ smooth turning
            (speed > 120 and self.angle_trend > 0.015)  # fast sweeper
        ):
            return True

        return False

    # ==========================
    # GENERATE COMMENTARY
    # ==========================
    def generate_commentary(self, S):

        speed = S.get("speedX", 0)

        prompt = f"""
You are a Formula 1 commentator.

The car just entered a corner at {speed:.0f} km/h.

Use a DIFFERENT style each time:
- aggressive
- dramatic
- analytical
- excited

Say ONE short exciting line.
Maximum 8 words.
"""

        commentary = self.granite_model.generate(prompt)
        commentary = re.sub(r'[\n"]', '', commentary).strip()

        return commentary

    # ==========================
    # UPDATE LOOP
    # ==========================
    def update(self):

        S = self.read_telemetry()
        if not S:
            return

        now = time.time()
        is_corner = self.detect_corner(S)

        # ✅ Trigger ONLY when entering a corner
        if is_corner and not self.in_corner:

            if now - self.last_corner_time > self.cooldown:
                self.last_corner_time = now
                self.in_corner = True

                threading.Thread(
                    target=self._async_commentary,
                    args=(S,),
                    daemon=True
                ).start()

        # Reset when exiting corner
        if not is_corner:
            self.in_corner = False

    # ==========================
    # ASYNC COMMENTARY
    # ==========================
    def _async_commentary(self, S):

        commentary = self.generate_commentary(S)

        if commentary:
            print("🎙", commentary)
            self.latest_comment = commentary
