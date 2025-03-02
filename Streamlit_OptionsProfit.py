import os
import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

strikeRatio = 1.05

def norm_cdf(x):
    """
    Calculates the CDF of the standard normal distribution at x.
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, T, and sigma must be greater than 0.")

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.lower() == 'call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type.lower() == 'put':
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return price

def preprocess_data(file):
    df = pd.read_csv(file)
    df = df.sort_index(ascending=True).iloc[::-1].reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    for col in ["Close/Last", "Open", "High", "Low"]:
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
    return df

def preprocess_risk_free_rate(file):
    rf_df = pd.read_csv(file)
    rf_df["observation_date"] = pd.to_datetime(rf_df["observation_date"])
    rf_df.set_index("observation_date", inplace=True)
    rf_df["DGS10"] = rf_df["DGS10"].fillna(method='ffill') / 100  # Convert to decimal
    return rf_df

def get_risk_free_rate(date, rf_df):
    if date in rf_df.index:
        return rf_df.loc[date, "DGS10"]
    else:
        return rf_df.iloc[rf_df.index.get_loc(date, method='nearest')]["DGS10"]

def volatility(index, df):
    if index < 252:
        raise ValueError("Index too low for trailing volatility calculation")
    prices = df['Close/Last'][index-252:index]
    log_returns = (prices / prices.shift(1)).apply(math.log).dropna()
    return log_returns.std() * math.sqrt(252)

def diffCheck(start_index, stock_file, risk_free_file):
    df = preprocess_data(stock_file)
    rf_df = preprocess_risk_free_rate(risk_free_file)
    row_count = len(df)
    netReturn, totalCost = 0, 0
    
    for i in range(start_index, row_count - 252):
        S, K = df['Close/Last'][i], df['Close/Last'][i] * strikeRatio
        vol = volatility(i, df)
        r = get_risk_free_rate(df['Date'][i], rf_df)
        callPrice = black_scholes_price(S, K, 1, r, vol, 'call')
        yearlyProfit = df['Close/Last'][i+252] - K
        netReturn += max(yearlyProfit, 0)
        totalCost += callPrice
    
    years = ((row_count - 252) - start_index) / 252
    return {
        "Total Profit": netReturn - totalCost,
        "Total Cost": totalCost,
        "ROI": (netReturn - totalCost) / totalCost,
        "Years": years,
        "Annualized Returns": (netReturn / totalCost) ** (1/years)
    }

def plot_call_price_vs_profit(stock_file, risk_free_file):
    df = preprocess_data(stock_file)
    rf_df = preprocess_risk_free_rate(risk_free_file)
    row_count = len(df)
    call_prices, yearly_profits = [], []

    for i in range(252, row_count - 252):
        S, K = df['Close/Last'][i], df['Close/Last'][i] * strikeRatio
        vol = volatility(i, df)
        r = get_risk_free_rate(df['Date'][i], rf_df)
        call_price = black_scholes_price(S, K, 1, r, vol, 'call')
        profit = df['Close/Last'][i+252] - K
        call_prices.append(call_price)
        yearly_profits.append(profit)
    
    fig, ax = plt.subplots()
    ax.scatter(call_prices, yearly_profits, alpha=0.7)
    ax.set_xlabel("Call Price")
    ax.set_ylabel("Yearly Profit")
    ax.set_title(f"Call Price vs. Yearly Profit for Uploaded Stock Options")
    ax.grid(True)
    return fig

def plot_time_series(stock_file, risk_free_file):
    df = preprocess_data(stock_file)
    rf_df = preprocess_risk_free_rate(risk_free_file)
    row_count = len(df)
    dates, call_prices, yearly_profits = [], [], []

    for i in range(252, row_count - 252):
        dates.append(df['Date'][i])
        S, K = df['Close/Last'][i], df['Close/Last'][i] * strikeRatio
        vol = volatility(i, df)
        r = get_risk_free_rate(df['Date'][i], rf_df)
        call_prices.append(black_scholes_price(S, K, 1, r, vol, 'call'))
        yearly_profits.append(df['Close/Last'][i+252] - K)
    
    fig, ax = plt.subplots()
    ax.plot(dates, call_prices, label='Call Price', color='blue')
    ax.plot(dates, yearly_profits, label='Yearly Profit', color='green')
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("Call Price and Yearly Profit Over Time")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    return fig

def cumulative_plot(start_index, stock_file, risk_free_file):
    df = preprocess_data(stock_file)
    rf_df = preprocess_risk_free_rate(risk_free_file)
    row_count = len(df)
    cumulative_cost, cumulative_profit, cumulative_gain = [], [], []
    total_cost, net_return = 0, 0
    dates = []

    for i in range(start_index, row_count - 252):
        S, K = df['Close/Last'][i], df['Close/Last'][i] * strikeRatio
        vol = volatility(i, df)
        r = get_risk_free_rate(df['Date'][i], rf_df)
        call_price = black_scholes_price(S, K, 1, r, vol, 'call')
        yearly_profit = df['Close/Last'][i+252] - K
        net_return += max(yearly_profit, 0)
        total_cost += call_price
        
        dates.append(df['Date'][i])
        cumulative_cost.append(total_cost)
        cumulative_profit.append(net_return - total_cost)
        cumulative_gain.append(net_return)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, cumulative_cost, label='Cumulative Cost', color='red')
    ax.plot(dates, cumulative_gain, label='Cumulative Gain', color='blue')
    ax.plot(dates, cumulative_profit, label='Cumulative Profit', color='green')
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("Cumulative Costs, Gains, and Profits Over Time")
    ax.legend()
    ax.grid()
    plt.xticks(rotation=45)
    return fig



st.sidebar.title("Black-Scholes Option Pricing Model")
st.sidebar.write("This app calculates the profitability of options using the Black-Scholes model.")
st.sidebar.write("In order to give it data on a stock, download CSV from Nasqdaq Historical Data (Minimum time length of 2 years).")
st.sidebar.link_button("Nasdaq Historical Data", "https://www.nasdaq.com/market-activity/stocks/smp/historical")
st.sidebar.write("Risk-free rate data is required for the calculations, the program has values from 01-02-1980 to 02-24-2025")
st.sidebar.write("If you need data from outside this time zone, please download the CSV.")
st.sidebar.link_button("FRED Risk-Free Rate Data", "https://fred.stlouisfed.org/series/DGS10")

linkedin_url = "www.linkedin.com/in/harsha-matta-4b67b22a1"
st.sidebar.divider()


st.sidebar.markdown("Created by:")
st.sidebar.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Harsha Matta`</a>', unsafe_allow_html=True)


stock_file = st.file_uploader("Upload the stock data CSV file", type=["csv"])
risk_free_file = st.file_uploader("Upload the risk-free rate data CSV file", type=["csv"])

if st.button("Run Analysis") and stock_file is not None and risk_free_file is not None:
    results = diffCheck(252, stock_file, risk_free_file)
    st.write("### Results")
    st.write(f"Total Profit: {results['Total Profit']}")
    st.write(f"Total Cost: {results['Total Cost']}")
    st.write(f"ROI: {results['ROI']:.2%}")
    st.write(f"Years: {results['Years']:.2f}")
    st.write(f"Annualized Returns: {results['Annualized Returns']:.2%}")

    st.write("### Call Price vs. Yearly Profit")
    st.pyplot(plot_call_price_vs_profit(stock_file, risk_free_file))

    st.write("### Call Price and Yearly Profit Over Time")
    st.pyplot(plot_time_series(stock_file, risk_free_file))
    
    st.write("### Cumulative costs, gains, and profits over time")
    st.pyplot(cumulative_plot(252, stock_file, risk_free_file))
else:
    st.error("Please upload both the stock data CSV file and the risk-free rate data CSV file.")
