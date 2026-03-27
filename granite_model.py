import ollama


class GraniteModel:
    def __init__(self, model_name="granite4:3b"):
        self.model_name = model_name
        print(f"Connecting to Ollama with {model_name}...")

        # Test connection
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt="Say Ready and nothing else.",
                options={"num_predict": 5}
            )
            print("✓ Model ready!")
        except Exception as e:
            print(f"✗ Ollama error: {e}")
            print("Make sure Ollama is running and the model is pulled!")
            raise

    def generate(self, prompt):
        result = ollama.generate(
            model=self.model_name,
            prompt=prompt
        )

        return result["response"]
