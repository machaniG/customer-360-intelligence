# AI-powered Decision System for Revenue Prioritization

## Problem

Businesses often build machine learning models like churn prediction or customer lifetime value (CLV), but these outputs rarely translate into clear business actions.

Teams are left asking:
- Which customers should we prioritize?
- What is the financial impact of losing them?
- What action should we take?

The gap: **predictions → decisions**

---

## Solution

I built an end-to-end decision system that combines multiple ML models into a unified business-facing layer.

The system:

- predicts churn probability  
- estimates customer lifetime value  
- segments customers by behavior  
- calculates **revenue at risk (Churn × CLV)**  

This enables **value-based prioritization of customers**

---

## Approach

### 1. Feature Engineering Pipeline

- Built a reusable pipeline to generate customer-level features from transactional data  
- Ensures consistency across all models  
- Designed for periodic scoring (e.g., monthly scoring) and retraining when needed 


### 2. Predictive Models

- **Churn Model** → classification  
- **CLV Model** → regression  
- **Segmentation** → clustering (K-Means)  

Each model serves a specific role in decision-making.


### 3. Decision Layer

Combined outputs into:

**Revenue at Risk (RaR)**  = Churn Probability × Predicted CLV  

This creates a **single prioritization metric** aligned with business goals.

---

### 4. AI Interface (LLM Agent)

Built a natural language interface on top of the data:

Example query:
> “Which customers should I contact to save the most revenue?”

Output includes:
- ranked customers  
- revenue impact  
- behavioral insights  
- recommended actions  

---

## Results / Impact

- Identifies high-value customers at risk of churn  
- Prioritizes actions based on financial impact  
- Reduces reliance on manual analysis  
- Makes ML outputs accessible to non-technical stakeholders  

Shifts focus from **model accuracy → business value**

---

## Key Highlights

- End-to-end ML system (not just a model)
- Modular feature pipeline (production mindset)
- Business-driven metric (Revenue at Risk)
- LLM integration for decision support
- Strong alignment with real-world use cases

---

## Tech Stack

- Python (Pandas, Scikit-learn)
- PostgreSQL (centralized data storage)
- LangChain (agent orchestration and tool-based querying over structured data)
- LLM APIs (OpenAI / Anthropic / local models)

---

## Trade-offs & Practical Considerations

### CLV Model Limitations

The Gamma-Gamma model used for CLV estimation assumes independence between purchase frequency and monetary value.

However, the dataset contains heterogeneous customer behavior:

- High-frequency, low-spend customers  
- Low-frequency, high-spend customers  

This violates model assumptions and leads to some **unrealistic or negative CLV predictions**.


### Design Approach

Rather than over-optimizing the CLV model, the focus was on building a **robust decision system**:

- Combining multiple models into a unified decision layer  
- Prioritizing customers using **revenue at risk (Churn × CLV)**  
- Ensuring outputs remain actionable despite imperfect predictions  


### Practical Outcome

- Individual CLV predictions may be noisy  
- However, the system still effectively identifies **high-value at-risk customers**  
- Prioritization remains meaningful at the top end (where business impact matters most)  


### Key Takeaway

In real-world systems:

**Imperfect models within a strong decision framework often deliver more value than perfectly optimized standalone models**


### Extensibility

The system is designed to evolve:

- Models are decoupled from the feature pipeline  
- Components can be replaced without affecting the overall system  

Examples:
- Replace Gamma-Gamma with ML-based or survival models  
- Upgrade Random Forest → Gradient Boosting (XGBoost, LightGBM)  

This reflects a **production-oriented, modular design mindset**

...

## Future Work

- Real-time scoring pipeline  
- A/B testing for retention strategies  
- Integration with CRM systems  