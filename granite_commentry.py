import time
import threading
import re


class LiveCommentator:

    def __init__(self, granite_model, cooldown=4):

        self.granite_model = granite_model
        self.cooldown = cooldown

        self.previous_state = None
        self.last_corner_time = 0
        self.in_corner = False


    # ==========================
    # TURN DETECTION (MAP DATA)
    # ==========================
    def detect_corner(self, S):

        track = S.get("track", [200]*19)
        angle = S.get("angle", 0)

        left = track[0]
        right = track[18]

        curvature = right - left

        # detect left/right curve
        if abs(curvature) > 15 or abs(angle) > 0.08:
            return True

        return False


    # ==========================
    # GENERATE COMMENTARY
    # ==========================
    def generate_commentary(self, S):

        speed = S.get("speedX", 0)
        angle = abs(S.get("angle", 0))

        prompt = f"""
You are a Formula 1 commentator.

The car just entered a corner.

Speed: {speed:.0f} km/h
Angle: {angle:.2f}

Say ONE short exciting line reacting to the corner.
Maximum 12 words.
"""

        commentary = self.granite_model.generate(prompt)

        commentary = re.sub(r'[\n"]', '', commentary).strip()

        return commentary


    # ==========================
    # UPDATE LOOP
    # ==========================
    def update(self, S):

        now = time.time()

        is_corner = self.detect_corner(S)

        if is_corner and not self.in_corner and now - self.last_corner_time > self.cooldown:

            self.in_corner = True
            self.last_corner_time = now

            threading.Thread(
                target=self._async_commentary,
                args=(S,),
                daemon=True
            ).start()

        if not is_corner:
            self.in_corner = False


    # ==========================
    # ASYNC COMMENTARY
    # ==========================
    def _async_commentary(self, S):

        commentary = self.generate_commentary(S)

        if commentary:
            print("🎙", commentary)
