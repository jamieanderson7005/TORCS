from race_engineer.granite_model import GraniteRaceEngineer
import time

print("Loading model...")
engineer = GraniteRaceEngineer()

print("\nTesting generation...")
start = time.time()

response = engineer.generate("You are an F1 race engineer. Say 'Copy that, pushing hard.' and nothing else.", max_tokens=20)

elapsed = time.time() - start
print(f"\nResponse: {response}")
print(f"Time taken: {elapsed:.1f} seconds")