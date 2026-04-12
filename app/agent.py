# app/agent.py
from agents.agent import build_agent
from scripts.ask_agent import _extract_agent_answer, run_interactive

def ask_agent(question: str) -> str:
    """Ask a single question and return the answer"""
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return _extract_agent_answer(result)

def run_agent():
    """Run the agent in interactive mode."""
    run_interactive()   
# Example usage:
"""
# To ask a single question:
python -m app.agent --question "What is the churn risk for customer 123?"   
# To start the interactive agent:
python -m app.agent
    # Then you can ask questions like:
    • Which 5 customers should I contact to save most revenue? 
    • What is the CLV for customer 456?
    • Show me all high-risk customers with CLV above £200   
    ... and so on. The agent will respond to each question based on the data it has access to and its reasoning capabilities. 
    You can exit the interactive mode by typing "exit" or "quit". 
    Make sure to have the necessary data and models set up for the agent to function properly, as it relies on those to provide accurate answers.
"""