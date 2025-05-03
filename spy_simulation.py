import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Constants
CSV_PATH = "data/spy500_history.csv"
INITIAL_DEPOSIT = 100000
DEFAULT_ANNUAL_RETURN = 0.10
DEFAULT_DIVIDEND_RATIO = 0.02
LAST_DATA_YEAR = 2025
END_YEAR = 2050

st.title("S&P 500 Investment Simulator")
st.write("This is a simulation of the S&P 500 investment growth, assume 100 k initial investment, show the balance (including dividend) and the total return (including dividend) of the investment over time.")

# User input
start_year = st.number_input(
    "Enter the starting year (between 1924 and 2025):",
    min_value=1924, max_value=END_YEAR, value=1990, step=1
)

# Load data
df = pd.read_csv(CSV_PATH)
df = df[["Year", "Annual_Return", "Dividend_Ratio"]]
df = df.dropna(subset=["Year"]).copy()
df["Year"] = df["Year"].astype(int)

# Prepare simulation years
years = list(range(start_year, min(2025, END_YEAR + 1)))  # Only up to 2024

# Build a lookup for returns
returns = {}
for _, row in df.iterrows():
    year = int(row["Year"])
    returns[year] = {
        "annual_return": float(row.get("Annual_Return", 0)) / 100,
        "dividend_ratio": float(row.get("Dividend_Ratio", 0)) / 100,
    }

# Simulation
balance = INITIAL_DEPOSIT
results = []
for year in years:
    if year in returns and year <= LAST_DATA_YEAR:
        annual_return = returns[year]["annual_return"]
        dividend_ratio = returns[year]["dividend_ratio"]
    else:
        annual_return = DEFAULT_ANNUAL_RETURN
        dividend_ratio = DEFAULT_DIVIDEND_RATIO
    total_return = annual_return + dividend_ratio
    balance = balance * (1 + total_return)
    results.append({
        "Year": year,
        "Annual Return": f"{annual_return*100:.2f}%",
        "Dividend Ratio": f"{dividend_ratio*100:.2f}%",
        "Total Return": f"{total_return*100:.2f}%",
        "Balance": balance
    })

results_df = pd.DataFrame(results)



# Plot with Seaborn
st.subheader("Balance Over Time")
sns.set_theme(style="whitegrid")

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot Balance on left Y-axis
sns.lineplot(
    data=results_df,
    x="Year",
    y="Balance",
    marker="o",
    ax=ax1,
    label="Balance",
    color="tab:blue"
)

ax1.set_xlabel("Year", fontsize=14)
ax1.set_ylabel("Balance ($K)", color="tab:blue", fontsize=14)
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x/1000:,.0f}K'))
ax1.tick_params(axis='y', labelcolor="tab:blue", labelsize=14)
ax1.tick_params(axis='x', labelsize=14)

# Create a second y-axis for Total Return
ax2 = ax1.twinx()
# Convert 'Total Return' from string percentage to float
results_df["Total Return %"] = results_df["Total Return"].str.rstrip('%').astype(float)
sns.lineplot(
    data=results_df,
    x="Year",
    y="Total Return %",
    marker="s",
    ax=ax2,
    label="Total Return",
    color="tab:red"
)
ax2.set_ylabel("Total Return (%)", color="tab:red", fontsize=14)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.tick_params(axis='y', labelcolor="tab:red", labelsize=14)

# Title and legend
fig.suptitle("S&P 500 Investment Growth and Annual Total Return", fontsize=16)
fig.tight_layout()
st.pyplot(fig)

# Show table
st.subheader("Simulation Results")
st.dataframe(results_df.style.format({"Balance": "${:,.2f}"}), use_container_width=True)