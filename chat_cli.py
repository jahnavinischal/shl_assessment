import requests
import json

API_URL = "http://localhost:8000/chat"

history = []

print("\nSHL Assessment Assistant")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Add user message
    history.append({
        "role": "user",
        "content": user_input
    })

    try:
        response = requests.post(
            API_URL,
            json={"messages": history}
        )

        data = response.json()

        assistant_reply = data["reply"]

        print(f"\nAssistant: {assistant_reply}\n")

        # Print recommendations separately
        if data.get("recommendations"):
            print("Recommendations:\n")

            for i, rec in enumerate(data["recommendations"], 1):
                print(f"{i}. {rec['name']}")
                print(f"   Type: {rec['test_type']}")
                print(f"   URL: {rec['url']}")

                if rec.get("duration"):
                    print(f"   Duration: {rec['duration']}")

                if rec.get("keys"):
                    print(f"   Keys: {', '.join(rec['keys'])}")

                print()

        # Store assistant reply in history
        history.append({
            "role": "assistant",
            "content": assistant_reply
        })

        # Optional: stop automatically
        if data.get("end_of_conversation"):
            print("Conversation ended.")
            break

    except Exception as e:
        print(f"\nError: {e}\n")