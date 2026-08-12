"""
Text-first testing entry point. No Twilio, no Pipecat, no audio — just
typing into a terminal and reading replies, per the build plan (Step 1).

Run from langgraph_app/:
    python run_text.py

Requires OPENAI_API_KEY set as an environment variable.
"""
from langchain_core.messages import HumanMessage
from graph import graph

def main():
    print("Health1st Clinic — text-first test mode. Type 'quit' to exit.\n")
    state = {
        "messages": [],
        "intent": None,
        "patient_name": None,
        "provider": None,
        "appointment_type": None,
        "requested_datetime": None,
        "contact_info": None,
        "target_appointment_id": None,
        "missing_slots": [],
        "awaiting_confirmation": False,
        "out_of_scope_flag": False,
    }

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = graph.invoke(state)

        # print the latest AI reply
        last_ai = state["messages"][-1]
        print(f"Agent: {last_ai.content}\n")


if __name__ == "__main__":
    main()
