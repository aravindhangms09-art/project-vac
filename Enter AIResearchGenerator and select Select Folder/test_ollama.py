
import requests

url = "http://localhost:11434/api/generate"

data = {
    "model": "gemma2:2b",
    "prompt": "Suggest three innovative research ideas about Artificial Intelligence in Education. Explain each idea briefly.",
    "stream": False
}

try:
    print("Connecting to Ollama...")

    response = requests.post(
        url,
        json=data,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    print("\nAI Research Ideas:\n")
    print(result["response"])

except requests.exceptions.RequestException as error:
    print("Connection error:", error)

except (ValueError, KeyError) as error:
    print("Response error:", error)

