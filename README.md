# Customer Retention Decision System

Focus retention efforts on the customers most likely to leave and cost you money.

## Overview

This is a customer retention system that **bridges the gap between CRM analytics and action by turning customer data into clear, prioritized retention decisions**. It can help businesses prevent revenue loss by identifying which customers are most valuable and most likely to leave. The system ranks customers based on expected value loss (revenue at risk) and highlights who to contact first and what to offer them to maximize impact. 

- **Targeting the top 20% of customers ranked by revenue-at-risk (using this system) captures ~52% of total revenue loss in the following three months, significantly outperforming churn-only (14%) and value-only (43%) approaches.**

- The system works directly from raw transaction data, automatically transforming it into customer-level insights, removing the need for manual data preparation. 
- It highlights who to contact first and why (high-priority customers and key insights about their behavior)
- Suggests targeted retention actions (e.g., personalized offers based on past purchases). 
- Results are delivered through a simple user interface chatbot and scheduled reports, making it easy for teams to act quickly without manual analysis.
- It is designed to stay up to date as new data arrives, allowing teams to easily refresh customer scores and keep retention efforts aligned with the latest behavior.
- Designed as a modular system that can adapt to different data environments, allowing performance to scale with the quality and richness of available customer data.


## Problem 

CRMs generate churn scores and even dashboards but:
- No clear who do we act on today?
- No prioritization tied to actual revenue impact
- Insights sit in dashboards and no action taken

- **My project fills this gap of turning data into clear, prioritized and repeatable actions**. 

My project bridges this gap by:

- Combining models into a single decision metric (revenue at risk)
- Produces a ranked list of customers to act on now
- Suggests concrete actions
- Delivers it automatically and repeatedly

The system goes beyond traditional CRM analytics to translate customer data into a prioritized list of actions, therefore addressing the common gap between CRM insights and execution.

## Impact (Evaluation)

On historical data:

- Targeting the top 20% of customers identified by the system captures ~52% of total revenue loss
This outperforms:
        - Using customer value alone (43%)
        - Using churn prediction alone (14%)

- This means businesses can save more revenue with the same effort by focusing on the right customers.

![Cumulative capture curve](artifacts/assets/evaluation.png)

**Insights**

- The combined revenue-at-risk model consistently outperforms bothe value-only and churn-only approaches, capturing the highest proportion of revenue loss across all customer segments.
- Customer value provides the primary signal while churn probability improves prioritization by identifying which high-value customers are at immediate risk, leading to better targeting efficiency.



**Baseline comparisons**

- Churn-only captures 14% of total revenue 
- Value-only captures 43% of total revenue 
- Combined approach captures 52% of revenue 

- The +9 percentage lift on total loss, which is ~21% relative improvement over CLV-only implies that when teams can only contact to 5-20% of customers, more revenue is saved with the same budget.

- While value explains most of the signal, churn adds critical timing information that improves targeting efficiency.

- Results reflect the characteristics of the dataset where customer value is a stronger predictor than churn. The system is designed to incorporate richer behavioral signals, and performance is expected to improve in environments with more informative churn patterns.


## Live Demo

Watch the system below:

[▶️ Demo Video](https://www.loom.com/share/b88a7162a7d149e6946e60b30c0bf962)


## Limitations

- Results depend on data quality and available signals
- In this dataset, customer value was a stronger driver and churn adds incremental improvement. In other settings with stronger churn signals, the uplift could be larger.
- Performance is expected to improve with richer behavioral data
- CLV is approximated using predicted purchase frequency and historical AOV, which introduces bias in absolute values. However, since the business objective is prioritization rather than exact forecasting, I evaluated the system based on its ability to rank customers by revenue-at-risk.


