import ollama
import time

print("Testing Ollama connection...")

start = time.time()
response = ollama.generate(
    model="granite3.3:2b",
    prompt="You are an F1 race engineer. Say 'Copy that, pushing hard.' and nothing else.",
    options={"num_predict": 20}
)
elapsed = time.time() - start

print(f"Response: {response['response']}")
print(f"Time taken: {elapsed:.1f} seconds")