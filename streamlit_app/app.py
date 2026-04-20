import streamlit as st
import requests
import pandas as pd


# CONFIG

API_BASE = "http://127.0.0.1:8000"
TIMEOUT = 5

# functions to fetch data with caching
@st.cache_data(ttl=60)
def get_revenue_data():
    response = requests.get(f"{API_BASE}/revenue-at-risk/topN", timeout=TIMEOUT)
    return response.json()

@st.cache_data(ttl=60)
def get_top_customers():
    response = requests.get(f"{API_BASE}/top-customers", timeout=TIMEOUT)
    return response.json()

@st.cache_data(ttl=60)
def get_revenue_summary():
    response = requests.get(f"{API_BASE}/revenue-at-risk/summary", timeout=TIMEOUT)
    return response.json()["data"]

# Check if backend is running
try:
    res = requests.get(f"{API_BASE}/docs", timeout=TIMEOUT)
    if res.status_code != 200:
        st.error("Backend API is not responding properly.")
        st.stop()
except Exception:
    st.error("🚨 FastAPI backend is not running. Start it with:\n\nuvicorn app.main:app --reload")
    st.stop()

st.set_page_config(
    page_title="Customer 360 Intelligence",
    layout="wide"
)

# start app

st.title("📊 Customer 360 Decision Intelligence")


# SIDEBAR NAVIGATION

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Customer Explorer", "AI Agent"]
)


# DASHBOARD

if page == "Dashboard":

    st.header("📊 Revenue Intelligence Dashboard")

    
    # FILTERS 
    
    st.sidebar.subheader("Filters")

    risk_threshold = st.sidebar.slider(
        "Minimum Revenue at Risk (£)",
        min_value=0,
        max_value=100000,
        value=1000,
        step=500
    )

    # Fetch data
    try:
        data = get_revenue_data() # top 10 for chart
        top10_risk_df = pd.DataFrame(data["data"])
        summary = get_revenue_summary() # aggregated total
        df_summary = pd.DataFrame([summary])

        data2 = get_top_customers()
        df_top = pd.DataFrame(data2["data"])

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    
    # SEGMENT FILTER 
   
    if "segment_label" in top10_risk_df.columns:
        segments = ["All"] + sorted(top10_risk_df["segment_label"].dropna().unique().tolist())

        selected_segment = st.sidebar.selectbox("Segment", segments)

        if selected_segment != "All":
            top10_risk_df = top10_risk_df[top10_risk_df["segment_label"] == selected_segment]
            df_top = df_top[df_top["segment_label"] == selected_segment]

    
    # RISK FILTER 
   
    top10_risk_df = top10_risk_df[top10_risk_df["revenue_at_risk"] >= risk_threshold]

    
    # KPI ROW
    total_risk = df_summary["total_revenue_at_risk"].sum()
    avg_clv = df_summary["avg_clv"].mean()
    customer_count = df_summary["total_customers"].sum()
   
    top10_risk = top10_risk_df["revenue_at_risk"].sum()
    
    
    col1, col2, col3 = st.columns(3)

    col1.metric("💸 Revenue at Risk", f"£{summary['total_revenue_at_risk']:,.0f}")
    col2.metric("⚠️ Customers at Risk", summary['total_customers'])
    col3.metric("📈 Avg CLV", f"£{summary['avg_clv']:,.0f}")

    st.markdown("---")

    # Calculate concentration
    top10_sum = top10_risk_df["revenue_at_risk"].sum()
    concentration = (top10_sum / summary['total_revenue_at_risk']) * 100
    st.info(f"📌 Top 10 customers account for **{concentration:.1f}%** of total revenue at risk")

    
    # CHART
    
    st.subheader("Top Customers by Revenue at Risk")

    chart_df = top10_risk_df.sort_values(
        by="revenue_at_risk", ascending=False
    ).head(10)

    st.bar_chart(
        chart_df.set_index("customer_id")["revenue_at_risk"], color= "#1b93b7"
    )

    
    # TABLES
   
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Revenue at Risk")
        st.dataframe(chart_df, use_container_width=True)

    with col2:
        st.subheader("Priority Customers")
        st.dataframe(df_top.head(10), use_container_width=True)

    
    # RUN SCORING BUTTON 
   
    st.markdown("---")
    st.subheader("⚙️ Run Monthly Scoring")

    score_date = st.date_input("Select scoring date")

    if st.button("Run Scoring"):

        with st.spinner("Running scoring pipeline..."):

            try:
                response = requests.post(
                    f"{API_BASE}/score-customers", 
                    params={"score_date": str(score_date)}
                )

                result = response.json()

                st.success("Scoring completed!")
                st.json(result)

                # clear cache to refresh data
                st.cache_data.clear()

            except Exception as e:
                st.error(f"Error: {e}")


# CUSTOMER EXPLORER

elif page == "Customer Explorer":

    st.header("👤 Customer Profile")

    customer_id = st.text_input("Enter Customer ID")

    if st.button("Fetch Profile"):

        try:
            response = requests.get(
                f"{API_BASE}/customer-profile/{customer_id}", timeout=TIMEOUT
            )
            data = response.json()

            if "data" in data:
                df = pd.DataFrame(data["data"])
                st.dataframe(df)
            else:
                st.warning("Customer not found")

        except Exception as e:
            st.error(f"Error: {e}")



# AI AGENT

elif page == "AI Agent":

    st.header("🤖 Ask the AI")

    question = st.text_input(
        "Ask a business question",
        placeholder="Which customers should I contact to save most revenue?"
    )

    if st.button("Ask"):

        with st.spinner("Thinking..."):

            try:
                response = requests.post(
                    f"{API_BASE}/ask-agent",
                    params={"question": question}
                )

                result = response.json()

                st.markdown("### 💡 Answer")
                st.write(result["response"])

            except Exception as e:
                st.error(f"Error: {e}")