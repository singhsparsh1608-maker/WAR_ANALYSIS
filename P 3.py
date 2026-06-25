import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Russia-Ukraine War Analysis",
    page_icon="⚔️",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        r"/Users/sparshsingh/Documents/GitHub/WAR ANALYSIS/DATA/PYTHON/CREA_of_PYHTON_SQL_READY.csv"
    )

    sql1 = pd.read_csv(
        r"/Users/sparshsingh/Documents/GitHub/WAR ANALYSIS/DATA/SQL/SQL_RES.11.xml.csv"
    )

    sql2 = pd.read_csv(
        r"/Users/sparshsingh/Documents/GitHub/WAR ANALYSIS/DATA/SQL/SQL_RES.22.xml.csv"
    )

    df.columns = [c.lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    df["day_number"] = np.arange(len(df))

    return df, sql1, sql2


df, sql1, sql2 = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Personnel Analysis",
        "Equipment Analysis",
        "Monthly Analysis",
        "Correlation Analysis",
        "Machine Learning",
        "Forecasting",
        "SQL Analysis",
        "Insights"
    ]
)

# -----------------------------
# OVERVIEW
# -----------------------------

if page == "Overview":

    st.title("Russia-Ukraine War Analysis Dashboard")

    st.markdown("""
    ### Project Overview

    This project combines:

    - Python Data Analysis
    - SQL Analytics
    - Exploratory Data Analysis
    - Machine Learning
    - Forecasting

    using Russia-Ukraine War datasets.
    """)

    col1, col2, col3, col4 = st.columns(4)

    # =========================
    # TOTAL RECORDS
    # =========================
    col1.metric("Total Records", len(df))

    # =========================
    # TOTAL PERSONNEL
    # =========================
    if "personnel" in df.columns:
        total_personnel = df["personnel"].sum()
        col2.metric("Total Personnel Loss", f"{total_personnel:,.0f}")
    else:
        col2.metric("Total Personnel Loss", "N/A")

    # =========================
    # TOTAL TANK
    # =========================
    if "tank" in df.columns:
        total_tank = df["tank"].sum()
        col3.metric("Total Tank Loss", f"{total_tank:,.0f}")
    else:
        col3.metric("Total Tank Loss", "N/A")

    # =========================
    # TOTAL DRONE
    # =========================
    if "drone" in df.columns:
        total_drone = df["drone"].sum()
        col4.metric("Total Drone Loss", f"{total_drone:,.0f}")
    else:
        col4.metric("Total Drone Loss", "N/A")

    # =========================
    # SLIDE DOWN TABLE
    # =========================
    with st.expander("📂 View Full Dataset (Click to Expand)"):

        st.dataframe(df, use_container_width=True)


# -----------------------------
# PERSONNEL ANALYSIS
# -----------------------------

elif page == "Personnel Analysis":

    st.header("Personnel Loss Trend")

    if "personnel" in df.columns:

        fig = px.line(
            df,
            x="date",
            y="personnel",
            title="Personnel Losses Over Time"
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# EQUIPMENT ANALYSIS
# -----------------------------

elif page == "Equipment Analysis":

    st.header("Equipment Loss Analysis")

    cols = []

    for c in [
        "tank",
        "aircraft",
        "helicopter",
        "drone",
        "artillery"
    ]:
        if c in df.columns:
            cols.append(c)

    if len(cols) > 0:

        totals = pd.DataFrame({
            "Equipment": cols,
            "Losses": [df[c].sum() for c in cols]
        })

        fig = px.bar(
            totals,
            x="Equipment",
            y="Losses",
            title="Equipment Loss Comparison"
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# MONTHLY ANALYSIS
# -----------------------------

elif page == "Monthly Analysis":

    st.header("Monthly Personnel Analysis")

    if "date" in df.columns and "personnel" in df.columns:

        monthly = (
            df.groupby(df["date"].dt.month)
            ["personnel"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            monthly,
            x="date",
            y="personnel",
            title="Average Personnel Loss by Month"
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# CORRELATION
# -----------------------------

elif page == "Correlation Analysis":

    st.header("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# MACHINE LEARNING
# -----------------------------

elif page == "Machine Learning":

    st.header("Linear Regression Model")

    if "personnel" in df.columns:

        X = df[["day_number"]]
        y = df["personnel"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        model = LinearRegression()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "MAE",
            round(mae, 2)
        )

        c2.metric(
            "R² Score",
            round(r2, 4)
        )

        result_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": predictions
        })

        fig = px.scatter(
            result_df,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted"
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# FORECASTING
# -----------------------------

elif page == "Forecasting":

    st.header("400-Day Forecast")

    if "personnel" in df.columns:

        X = df[["day_number"]]
        y = df["personnel"]

        model = LinearRegression()

        model.fit(X, y)

        future = pd.DataFrame({
            "day_number":
            np.arange(
                len(df),
                len(df) + 400
            )
        })

        future_pred = model.predict(future)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["day_number"],
                y=df["personnel"],
                name="Historical"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=future["day_number"],
                y=future_pred,
                name="Forecast"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #1f4e79, #2563eb);
                padding: 16px 18px;
                border-radius: 10px;
                color: white;
                font-family: Arial, sans-serif;
                font-size: 17px;
                line-height: 1.5;
                margin-top: 12px;
            ">
                <b>📊 Forecast Insight:</b><br><br>

                Forecast shows overall future trend of losses based on all past data, 
    
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# SQL ANALYSIS
# -----------------------------

elif page == "SQL Analysis":

    st.header("SQL Query Outputs")

    st.subheader("Monthly Personnel Query")

    st.dataframe(sql1)

    try:
        fig1 = px.bar(
            sql1,
            x=sql1.columns[0],
            y=sql1.columns[1]
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )
    except:
        pass

    st.subheader("Equipment Summary Query")

    st.dataframe(sql2)

    try:
        fig2 = px.bar(
            sql2,
            x=sql2.columns[0],
            y=sql2.columns[1]
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )
    except:
        pass

# -----------------------------
# INSIGHTS
# -----------------------------

elif page == "Insights":

    st.header("Key Findings")

    st.markdown("""
    ### Project Findings

    - Personnel losses show a strong upward trend.
    - Equipment losses increased significantly over time.
    - Tanks and drones are among the most impacted assets.
    - Correlation analysis reveals relationships among military metrics.
    - Machine Learning was used to model personnel loss trends.
    - Forecasting provides future projections based on historical data.

    ### Technologies Used

    - Python
    - Pandas
    - NumPy
    - SQL
    - Plotly
    - Streamlit
    - Scikit-Learn
    """)