# AI-powered Decision System for Revenue Prioritization

## Problem

Businesses generate churn scores and CLV predictions but struggle to:

- prioritize customers effectively
- quantify revenue impact
- translate insights into actions

The gap: **predictions → decisions**

---

## Solution

I built a **decision intelligence** system that:

- combines churn + CLV + segmentation
- computes **Revenue at Risk (RaR)**
- exposes insights via API and AI interface

This enables **value-based prioritization of customers**

---

## Approach

### 1. Feature Engineering Pipeline

- Built a reusable pipeline to generate customer-level features from transactional data  
- Ensures consistency across all models  
- Designed for periodic scoring (e.g., monthly scoring) and retraining when needed 


### 2. Predictive Models

- **Churn Model** → classification  
- **CLV Model** → Gamma-Gamma and BG/NBD models  
- **Segmentation** → clustering (K-Means)  

Each model serves a specific role in decision-making.


### 3. Decision Layer

Combined outputs into:

**Revenue at Risk (RaR)**  = Churn Probability × Predicted CLV  

This creates a **single prioritization metric** aligned with business goals.


### 4. AI Interface (LLM Agent)

Built a natural language interface on top of the data:

Example query:
> “Which customers should I contact to save the most revenue?”

Output includes:
- ranked customers  
- revenue impact  
- behavioral insights  
- recommended actions  


### 5. FastAPI & Streamlit UI

Built FastAPI service for scoring + querying and developed Streamlit app for business users. 

---

## Results 

- Identifies high-value customers at risk of churn  
- Prioritizes actions based on financial impact  
- Makes ML outputs accessible to non-technical stakeholders 

Enabled:
   - targeted retention strategies
   - revenue-based prioritization
   - faster decision-making 

---

## Impact

- Shifts focus from **model accuracy → business value**
- Shift from **descriptive analytics → decision intelligence**
- Reduces reliance on manual analysis
- Provides **clear, actionable recommendations**

---

## Key Highlights

- Slightly imperfect models inside a strong decision system delivers more value than a perfect model in isolation.

---

## Tech Stack

- **Python** (Pandas, Scikit-learn, Lifetimes)
- **FastAPI** (serving ML pipeline and agent endpoints)
- **PostgreSQL** (customer feature & scoring store)
- **Streamlit** (interactive decision dashboard)
- **LangChain** (LLM agent orchestration and tool-based querying over structured data)
- **LLM Providers** (OpenAI, Anthropic, Ollama)
- **Docker** (containerization-ready architecture)

---

## Trade-offs & Practical Considerations

### CLV Model Limitations

The Gamma-Gamma model used for CLV estimation assumes independence between purchase frequency and monetary value.

However, the dataset contains heterogeneous customer behavior:

- High-frequency, low-spend customers  
- Low-frequency, high-spend customers  

This violates model assumptions and leads to some **unrealistic or negative CLV predictions**.


### Design Approach

Rather than over-optimizing the CLV model, the focus was on building a **robust decision system** and the CLV prediction can be switched to a better model:

- Combining multiple models into a unified decision layer  
- Prioritizing customers using **revenue at risk (Churn × CLV)**  
- Ensuring outputs remain actionable despite imperfect predictions  


### Practical Outcome

- Individual CLV predictions may be noisy  
- However, the system still effectively identifies **high-value at-risk customers**  
- Prioritization remains meaningful at the top end (where business impact matters most)  


### Extensibility

The system is designed to evolve:

- Models are decoupled from the feature pipeline  
- Components can be replaced without affecting the overall system  

Examples:
- Replace Gamma-Gamma with ML-based or survival models  
- Upgrade Random Forest → Gradient Boosting (XGBoost, LightGBM)  

---

## Future Work

- RAGs for stock code and product description
- A/B testing for retention strategies  
- Integration with CRM systems 
- Real-time scoring pipeline  