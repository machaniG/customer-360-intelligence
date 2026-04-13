# AI-powered Decision System for Revenue Prioritization

An end-to-end customer intelligence system that transforms raw transactional data into **actionable revenue decisions**.

Instead of dashboards or isolated model outputs, this system answers a critical business question:

- *“Which customers should we act on right now to maximize revenue impact?”*

---

## Overview

Most businesses generate churn scores, CLV predictions, and segmentation outputs — but struggle to **translate them into action**.

This project bridges that gap by combining:

- Predictive modeling (churn, CLV, segmentation)
- Centralized data infrastructure (PostgreSQL)
- A natural language AI interface (LLM agent)

**The result:** a system that turns data into **clear, prioritized actions for revenue teams**

---

## Key Capabilities

### 1. Customer Prioritization Engine

Combines multiple models into a single decision layer:

- **Churn Risk** → likelihood of customer leaving  
- **CLV (Customer Lifetime Value)** → future revenue potential  
- **Segmentation** → behavioral context  
- **Revenue at Risk (RaR)** → prioritization metric  

- Enables **value-based actioning**, not just scoring


### 2. Centralized Intelligence Layer

- All customer features and model outputs stored in **PostgreSQL**
- Single source of truth for analytics and decision-making
- Designed for **repeatable monthly scoring pipelines**


### 3. AI-Powered Decision Interface

A LangChain-based LLM agent sits on top of the database and acts as a **strategic assistant**.

Instead of querying tables manually, users can ask:

- *“Which customers should I contact to save the most revenue?”*

And receive:

- prioritized customer list  
- revenue impact  
- behavioral context  
- recommended actions  

---

## Example Interaction

**Question:**
> Which 5 customers should I contact to save most revenue?

**Answer:**

```plaintext
🤖 Agent:
**Customer ID: 12678.0**
- **Churn Risk:** 44.8%
- **Predicted CLV (12m):** £185,193.69
- **Revenue at Risk:** £82,973.51
- **Segment:** At Risk
- **Behavioral Context:** Last purchase was 28 days ago with a monthly purchase rate of 0.8.
- **Recommended Action:** Immediate re-engagement discount or "we miss you" personalized email.

**Customer ID: 14062.0**
- **Churn Risk:** 47.2%
- **Predicted CLV (12m):** £161,930.68
- **Revenue at Risk:** £76,364.24
- **Segment:** At Risk
- **Behavioral Context:** Last purchase was 34 days ago with a monthly purchase rate of 0.72.
- **Recommended Action:** Immediate re-engagement discount or "we miss you" personalized email.

**Customer ID: 13680.0**
- **Churn Risk:** 51.6%
- **Predicted CLV (12m):** £129,844.68
- **Revenue at Risk:** £67,055.26
- **Segment:** At Risk
- **Behavioral Context:** Last purchase was 36 days ago with a monthly purchase rate of 0.81.
- **Recommended Action:** Immediate re-engagement discount or "we miss you" personalized email.

**Customer ID: 15369.0**
- **Churn Risk:** 44.5%
- **Predicted CLV (12m):** £109,536.42
- **Revenue at Risk:** £48,766.11
- **Segment:** Lost
- **Behavioral Context:** Last purchase was 57 days ago with a monthly purchase rate of 1.13.
- **Recommended Action:** Aggressive win-back offer or feedback survey to understand why they stopped buying.

**Customer ID: 13324.0**
- **Churn Risk:** 41.9%
- **Predicted CLV (12m):** £109,157.31
- **Revenue at Risk:** £45,801.18
- **Segment:** Lost
- **Behavioral Context:** Last purchase was 16 days ago with a monthly purchase rate of 1.66.
- **Recommended Action:** Aggressive win-back offer or feedback survey to understand why they stopped buying.
```

This poject moves from data analysis → decision support

---

## System Architecture
```plaintext
Raw Transaction Data
        ↓
Feature Engineering Pipeline (Reusable)
        ↓
ML Models
  - Churn Prediction
  - CLV Estimation
  - K-Means Segmentation
        ↓
PostgreSQL (Customer 360 Table)
        ↓
LLM Agent (Natural Language Interface)
        ↓
Actionable Business Insights
```
---

## ⚙️ Configurable LLM Layer

The system is model-agnostic and supports multiple providers for flexibility, cost control, and future-proofing:

### Supported Options

- **Local Models** (Ollama / LLaMA) → privacy + zero cost
- **OpenAI (GPT-4o)** → high performance
- **Anthropic (Claude)** → alternative cloud provider

Switch models via .env — no code changes required.

```bash
LLM_PROVIDER=local
LLM_MODEL=llama3.1

# or

LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```
---

## Features

- Customer prioritization based on revenue impact
- Revenue-at-risk estimation (Churn × CLV)
- Natural language querying of customer data
- Automated action recommendations
- Modular ML pipeline with reusable features
- Centralized data storage for scalability

---

## Why This Matters

This system does not stop at **dashboards and model outputs**, it goes further and:

✅ identifies who to act on
✅ quantifies business impact
✅ suggests what to do next

As a result, bridging the gap between data science and decision-making.

---

## Tech Stack

- **Python** (Pandas, Scikit-learn, Lifetimes)
- **PostgreSQL** (centralized data storage)
- **LangChain** (agent orchestration and tool-based querying over structured data)
- **LLM APIs** (OpenAI, Anthropic, Ollama)
- Feature engineering pipeline (modular design)

---

## ⚠️ Model Limitations & Design Choices

### CLV Modeling Assumptions

Customer Lifetime Value (CLV) is estimated using the Gamma-Gamma and BG/NBD models from the Lifetimes library.

The Gamma-Gamma model assumes: 
- monetary value is independent of purchase frequency  
- customers exhibit relatively stable purchasing behavior 

### Observed Data Behavior

In this dataset, customer behavior violates these assumptions:

- Some customers purchase **frequently but spend very little per transaction**  
- Others purchase **rarely but with very high transaction values**  

This creates inconsistency between frequency and monetary value that does not satisfy the model assumptions.


### Impact on Predictions

As a result:
- Some CLV estimates become **unrealistic or negative**
- The model struggles to generalize across different customer types  


### Design Decision

Instead of forcing model assumptions or overfitting:

- The model was retained as part of a broader decision system  
- High-value customers are still effectively ranked  
- Revenue-at-risk prioritization remains useful for business decisions  

The focus is on **decision quality, not perfect individual predictions**

---

## System Flexibility

The architecture is modular:

- Feature pipeline is shared across models  
- Models can be swapped independently  

Examples:
- Replace Gamma-Gamma with ML-based regression or survival models  
- Replace Random Forest with XGBoost or LightGBM  

This enables continuous improvement without redesigning the system

---

## Future Improvements

- Pricing and revenue optimization module
- A/B testing framework for retention strategies
- Real-time scoring pipeline
- API layer for integration with CRM tools

---
## Quick Usage

```python
# step 1. create a virtual environment
python -m venv env
source env/bin/activate

# step 2. install dependencies
pip install requirements.txt

# step 3. run training and scoring:
python scripts/full_pipeline.py
python scripts/score_combined.py or  python -m scripts.score_combined --score_date 2010-10-31 # use new score date for next month
#result:
combined_customer_scores.csv

# step 4. write results to postgres. Ensure credentials are in db/config.py
python -m scripts.store_scores \
    --scores_path outputs/combined_customer_scores.csv \
    --features_path outputs/customer_features.csv \
    --as_of_date 2011-10-31 # change as_of_date for next month

# step 5. ask the agent any question
python -m scripts.ask_agent --question "Which 5 customers should I contact to save most revenue?"

# evaluate churn scores
python models/churn/evaluate_scores.py --output_csv outputs/churn_evaluation.csv
This generates ROC AUC, PR AUC, confusion matrix metrics, and top-k lift analysis.
```
