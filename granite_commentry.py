import time
import threading
import re
from telemetry_client import FileTORCSClient


class LiveCommentator:

    def __init__(self, granite_model, telemetry_client=None):
        self.granite_model = granite_model

        # Cooldowns
        self.event_cooldown = 1.5
        self.flow_cooldown = 3.5

        self.last_event_time = 0
        self.last_event = None
        self.latest_comment = None

        # Previous state
        self.prev_speed = 0
        self.prev_angle = 0
        self.prev_trackpos = 0

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
    # EVENT DETECTION
    # ==========================
    def detect_event(self, S):

        speed = S.get("speedX", 0)
        angle = S.get("angle", 0)
        trackpos = S.get("trackPos", 0)

        # 🔥 MAJOR EVENTS (priority)
        if abs(trackpos) > 0.8:
            return "off_track"

        if self.prev_speed - speed > 20:
            return "braking"

        if speed - self.prev_speed > 15:
            return "acceleration"

        if abs(angle - self.prev_angle) > 0.05:
            return "turn_in"

        # 🌊 FLOW EVENTS (ambient commentary)
        if speed > 160:
            return "cruising_fast"

        if speed > 80:
            return "cruising"

        return None

    # ==========================
    # GENERATE COMMENTARY
    # ==========================
    def generate_commentary(self, S, event):

        speed = S.get("speedX", 0)

        prompt = f"""
You are a Formula 1 commentator.

Event: {event}
Speed: {speed:.0f} km/h

React specifically to the event.

Rules:
- One short line
- Max 8 words
- No punctuation at the end
- High energy, broadcast style
- Avoid repeating phrases

Examples:
acceleration: "Launches out like a rocket"
braking: "Huge stop right on the limit"
off_track: "He’s gone wide that’s costly"
turn_in: "Throws it in aggressively"
cruising_fast: "Flying down the straight at full speed"
cruising: "Maintaining strong pace through this section"

Now respond:
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
        event = self.detect_event(S)

        if event:

            # 🚫 Prevent boring repetition
            if event == self.last_event and event in ["cruising", "cruising_fast"]:
                pass
            else:
                # 🎯 Decide cooldown type
                is_major = event in ["off_track", "braking", "acceleration"]
                cooldown = self.event_cooldown if is_major else self.flow_cooldown

                if now - self.last_event_time > cooldown:
                    self.last_event_time = now
                    self.last_event = event

                    threading.Thread(
                        target=self._async_commentary,
                        args=(S, event),
                        daemon=True
                    ).start()

        # ✅ Update previous state
        self.prev_speed = S.get("speedX", 0)
        self.prev_angle = S.get("angle", 0)
        self.prev_trackpos = S.get("trackPos", 0)

    # ==========================
    # ASYNC COMMENTARY
    # ==========================
    def _async_commentary(self, S, event):

        commentary = self.generate_commentary(S, event)

        if commentary:
            print("🎙", commentary)
            self.latest_comment = commentary
