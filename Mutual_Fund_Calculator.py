import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("MUTUAL FUND SIP Calculator")

start_sip = st.number_input("Starting SIP (₹)", 500, 200000, 1000)
annual_return = st.slider("Annual Return (%)", 5, 25, 12)
step_up_percent = st.slider("Yearly SIP Increase (%)", 0, 50, 10)
years = st.slider("Investment Duration (Years)", 1, 40, 20)

def step_up_sip(start_sip, annual_return, years, step_up_percent):
    monthly_rate = annual_return / (12 * 100)
    months = years * 12
    sip = start_sip
    total_invested = 0
    portfolio = 0
    data = []

    for m in range(1, months + 1):
        if m % 12 == 1 and m != 1:
            sip *= (1 + step_up_percent / 100)

        total_invested += sip
        portfolio = (portfolio + sip) * (1 + monthly_rate)

        if m % 12 == 0:
            data.append([m//12, sip, total_invested, portfolio])

    return pd.DataFrame(
        data,
        columns=["Year", "Monthly SIP", "Total Invested", "Portfolio Value"]
    ), total_invested, portfolio


df, invested, value = step_up_sip(
    start_sip, annual_return, years, step_up_percent
)

st.metric("Total Invested", f"₹ {round(invested):,}")
st.metric("Final Value", f"₹ {round(value):,}")

st.dataframe(df)

plt.figure()
plt.plot(df["Year"], df["Total Invested"], label="Total Invested")
plt.plot(df["Year"], df["Portfolio Value"], label="Portfolio Value")
plt.legend()
st.pyplot(plt)