# ask_agent.py
# Interactive CLI for the Customer 360 Agent.
# Starts a conversation loop where you can ask natural language questions
# and the agent fetches data from PostgreSQL to answer them.
#
# Usage:
#   python ask_agent.py
#   python ask_agent.py --question "Which 5 customers should I contact to save most revenue?"
#   python ask_agent.py --verbose false    # hide the reasoning chain

import argparse
import logging
import sys
from pathlib import Path

from agents.agent import build_agent  
from dotenv import load_dotenv; load_dotenv()

load_dotenv(Path(__file__).parent / ".env")

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,          # suppress noisy library logs in the CLI
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "agent.log", mode="a"),
    ],
)
# Show our own agent logs in the file but not the console
logging.getLogger("agent").setLevel(logging.INFO)
logging.getLogger("db").setLevel(logging.INFO)

from agents.agent import build_agent


WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════╗
║          Customer 360 Intelligence Agent                 ║
║  Ask anything about churn risk, CLV or segments.        ║
║  Type 'exit' or 'quit' to end the session.              ║
╚══════════════════════════════════════════════════════════╝

Example questions:
  • Which 5 customers should I contact to save most revenue?
  • Show me all high-risk customers with CLV above £200
  • Give me a breakdown of customer segments
  • Tell me about customer 12345
  • Who are my most valuable customers that are about to churn?
"""


def _extract_agent_answer(result):
    """Robustly extract the text answer from LangChain's response object."""
    if isinstance(result, dict):
        # Handles 'output' key (standard) or the last message content
        return result.get("output") or result.get("messages", [])[-1].content
    return getattr(result, "content", str(result))

def run_interactive():
    """Starts a loop for continuous questioning."""
    print("--- Customer 360 Intelligence Agent ---")
    print("Type 'exit' or 'quit' to end.\n")
    agent = build_agent()
    
    while True:
        question = input("🔍 You: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
            
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
            print(f"\n🤖 Agent:\n{_extract_agent_answer(result)}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}")


def run_single(question: str, verbose: bool = True) -> str:
    """Ask a single question and return the answer. Useful for scripting."""
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return _extract_agent_answer(result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Customer 360 Intelligence Agent — ask questions in plain English."
    )
    parser.add_argument(
        "--question", "-q",
        default=None,
        help="Ask a single question and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--verbose",
        default="true",
        choices=["true", "false"],
        help="Show the agent's reasoning chain. Default: true.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    verbose = args.verbose.lower() == "true"

    if args.question:
        # Single-shot mode
        answer = run_single(args.question, verbose=verbose)
        print(f"\n🤖 Agent:\n{answer}\n")
    else:
        # Interactive loop
        run_interactive(verbose=verbose)

# Example usage:
"""
# steps to use local model:
# 1. install llama
# 2. in new terminal run the server: ollama serve
# 3. pull the model: ollama pull llama3.1
# 4. run the agent: python ask_agent.py --question "Which 5 customers should I contact to save most revenue?"
python -m scripts.ask_agent --question "Which 5 customers should I contact to save most revenue?"
# or just start the interactive agent:
python ask_agent.py
    # Then you can ask questions like:
    • Which 5 customers should I contact to save most revenue?
    • Show me all high-risk customers with CLV above £200       
    • Give me a breakdown of customer segments
    • Tell me about customer 12345
    • Who are my most valuable customers that are about to churn?
    • exit   # to quit the agent 
    stop the server with Ctrl+C in the terminal where it's running   
"""