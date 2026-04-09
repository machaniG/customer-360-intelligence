# agent/agent.py
#
# Customer 360 LangChain Agent
# ─────────────────────────────
# A ReAct agent that answers natural language questions about customer
# churn, CLV and segmentation by calling the DB query tools.
#
# The agent reasons step-by-step (Thought → Action → Observation) and
# combines tool outputs with business context to give clear, actionable answers.
#
# Supported LLM providers (set LLM_PROVIDER in .env):
#   openai    — GPT-4o (default)
#   anthropic — Claude 3.5 Sonnet

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent

from agents.tools import AGENT_TOOLS

load_dotenv(Path(__file__).parent.parent / ".env")
logger = logging.getLogger(__name__)

# ── LLM provider selection ─────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", None)

# prompt
SYSTEM_PROMPT = """You are a Customer Intelligence Agent for a retail business. 
You provide data-backed insights using churn_risk, CLV_12_month, revenue_at_risk, and behavioral features.

STRUCTURE RULES (CRITICAL):
For every customer, use this exact block format. Do not group recommendations.
**Customer ID: [customer_id]**
- **Churn Risk:** [churn_risk * 100]%
- **Predicted CLV (12m):** [CLV_12_month formatted as currency]
- **Revenue at Risk:** [revenue_at_risk formatted as currency]
- **Segment:** [segment_label]
- **Behavioral Context:** [Summary using purchase_rate_monthly and recency_days]
- **Recommended Action:** [Specific task based on segment and risk level]

STRICT MAPPING RULES:
- **CLV_12_month**: This is the predicted 12-month clv value. Always look for the "CLV_12_month" key in the tool output for this.
- **revenue_at_risk**: This is the potential value loss (e.g., £37,615.32). Always look for the "revenue_at_risk" key in the tool output for this.
- **customer_id**: Use the full ID provided.
- If a value looks suspicious (e.g., very low clv for a Platinum customer), re-read the tool output to find the correct "CLV_12_month" key.

ACTION LOGIC:
- Champions/Loyal: High-touch VIP outreach, early access to new products, or loyalty rewards.
- At-Risk: Immediate re-engagement discount, "we miss you" personalized email, or a phone call.
- Lost/Inactive: Aggressive win-back offer or feedback survey to understand why they stopped buying.
- High Spend/Low Frequency: Personal shopping assistance or reminders of new arrivals in their preferred category.

OPERATIONAL RULES:
1. Always fetch data before answering — never make up customer data.
2. Always call tools before answering questions.
3. Use 'get_priority_customers' for general "who should I contact" queries.
4. If a user asks for "High Risk", set risk_band='high'.
5. Ensure the "Recommended Action" is unique to the customer's data points before moving to the next customer.
6. Express CLV and Revenue at Risk as currency (e.g., £145.30).
7. Use the exact keys 'CLV_12_month' and 'revenue_at_risk' from the JSON response.
8. Every customer ID must have its own 'Recommended Action' bullet before the next ID.
"""


def _build_llm():
    """Instantiate LLM based on provider."""

    # ── Anthropic ──────────────────────────────────────────────────────────────
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model_name = LLM_MODEL or "claude-3-5-sonnet-20241022"

        model = ChatAnthropic(
            model=model_name,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0,
            max_tokens=4096,
        )
        logger.info("LLM: Anthropic %s", model_name)
        return model

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        model_name = LLM_MODEL or "gpt-4o"

        model = ChatOpenAI(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )
        logger.info("LLM: OpenAI %s", model_name)
        return model

    # ── Local (Ollama / Llama 3) ───────────────────────────────────────────────
    elif LLM_PROVIDER == "local":
        from langchain_ollama import ChatOllama

        model_name = LLM_MODEL or "llama3"

        model = ChatOllama(
            model=model_name,
            temperature=0,
        )
        logger.info("LLM: Local (Ollama) %s", model_name)
        return model

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")



# ── Agent factory ──────────────────────────────────────────────────────────────

def build_agent():
    """Build and return a LangChain agent using the new create_agent API."""
    llm = _build_llm()

    agent = create_agent(
        model=llm,
        tools=AGENT_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    logger.info("Customer 360 agent ready with %d tools", len(AGENT_TOOLS))
    return agent

