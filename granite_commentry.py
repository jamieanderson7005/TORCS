import socket
import sys
import getopt
import os
import time
import pyttsx3
import re
import threading
import queue
 

# ================= TTS & GRANITE IMPORTS =================
import pyttsx3

import torch
torch.set_num_threads(2) 

from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1.0)


device = "cpu"
granite_model_path = "ibm-granite/granite-3.1-1b-a400m-base"

tokenizer = AutoTokenizer.from_pretrained(granite_model_path)

granite_model = AutoModelForCausalLM.from_pretrained(
    granite_model_path,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
).to(device)

granite_model.eval()

def interpret_state(S):
    speed = S['speedX']
    angle = S['angle']
    pos = S['trackPos']

    if speed < 5:
        pace = "barely rolling forward"
    elif speed < 60:
        pace = "building speed steadily"
    else:
        pace = "flying down the circuit"

    if abs(angle) > 0.5:
        steering = "struggling for control"
    elif abs(angle) > 0.2:
        steering = "making small corrections"
    else:
        steering = "looking perfectly balanced"

    if abs(pos) > 1:
        track = "drifting dangerously wide"
    elif abs(pos) > 0.5:
        track = "running close to the edge of the track"
    else:
        track = "holding the ideal racing line"

    return pace, steering, track

def build_race_context(S):
    """
    Build a minimal, factual description of the current car and nearby opponents.
    Granite will turn this into live commentary itself.
    """
    context_lines = []

    # Basic info about our car
    context_lines.append(f"Your car is at speed {S['speedX']:.1f} and angle {S['angle']:.2f} on the track position {S['trackPos']:.2f}.")

    # Nearby opponents
    for idx, dist in enumerate(S['opponents']):
        if dist < 100:  # only relevant cars
            context_lines.append(f"Car #{idx+1} is {dist:.1f} meters away.")

    return " ".join(context_lines)

def generate_commentary(S):
    context = build_race_context(S)  # raw state only

    prompt = f"""
You are a passionate TORCS race commentator. 
Generate energetic, dynamic commentary in your own words. 
Talk only about the cars currently on the TORCS track.
Do NOT repeat phrases. 
Do NOT invent other cars or make generic F1 statements.
Do NOT mention telemetry numbers or technical data.
Focus on positions, speed, and interactions.

Current situation: {context}

Commentary:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = granite_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            top_k=50
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    commentary = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return commentary


# ================== HELPER FUNCTION TO CLEAN COMMENTARY ==================
def clean_commentary(raw_text):
    """
    Removes Granite system/user tokens and extra newlines.
    """
    clean_text = re.sub(r"<\|.*?\|>", "", raw_text)  # Remove <|start_of_role|> etc.
    clean_text = clean_text.replace("\n", " ").strip()  # Flatten lines
    return clean_text

# ================== THREADING FUNCTION FOR TTS ==================

# Global queue
tts_queue = queue.Queue()


def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak_async(text):
    tts_queue.put(text)
