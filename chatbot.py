"""Demo 1: a terminal chatbot running on a local Ollama model.

The `ollama` package is a thin wrapper over Ollama's HTTP API
(http://localhost:11434/api/chat), so this is the same as calling the API directly.

Run:  python chatbot.py
"""

import ollama

MODEL = "gemma3:1b"

# The model has no memory of its own: we resend the whole conversation every turn.
messages = [
    {"role": "system", "content": "You are a friendly, concise assistant."}
]

print(f"Chatting with {MODEL}. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit"):
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    # stream=True gives us the reply piece by piece, like ChatGPT's typing effect.
    print("Bot: ", end="", flush=True)
    reply = ""
    for part in ollama.chat(model=MODEL, messages=messages, stream=True):
        text = part["message"]["content"]
        print(text, end="", flush=True)
        reply += text
    print("\n")

    messages.append({"role": "assistant", "content": reply})
