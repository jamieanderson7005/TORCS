import time

from granite_model import GraniteModel
from granite_commentary import LiveCommentator

model = GraniteModel()
commentator = LiveCommentator(model)

print("🎙 Commentary system running...")

while True:
    commentator.update()
    commentator.latest_comment = None
    time.sleep(0.2)
