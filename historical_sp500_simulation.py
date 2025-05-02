import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better visualization
sns.set_style("whitegrid")
sns.set_palette("husl")

# Read and process S&P 500 historical data
def load_sp500_data():
    df = pd.read_csv('data/spy500_history.csv')
    df['Annual_Return'] = df['Total_Return'] / 100  # Convert to decimal
    return df

def get_historical_returns(start_year=1924, end_year=2020, 
simulation_years=30, default_return=0.10):
    """Get historical returns for the specified period, using default return when data is missing"""
    df = load_sp500_data()
    returns = []
    
    # Calculate how many years of returns we need
    needed_years = simulation_years + 1  # +1 because we need returns for the start year too
    
    for year in range(start_year, start_year + needed_years):
        year_data = df[df['Year'] == year]
        if not year_data.empty and not pd.isna(year_data['Annual_Return'].iloc[0]):
            returns.append(year_data['Annual_Return'].iloc[0])
        else:
            returns.append(default_return)
            print(f"Using default return of {default_return:.1%} for year {year}")
    
    return returns

def run_simulation(start_year, simulation_years=30):
    """Run a simulation for a given start year"""
    base_balance = 100000
    withdrawal_schedule = [
        12282, 13024, 13284, 14087, 14368, 15236, 15541, 16479, 16809, 17824, 18181,
        19278, 19664, 20852, 21269, 22562, 23013, 24412, 24900, 26414, 26943, 28581,
        29152, 30925, 31543, 33461, 34130, 36206, 36930, 39175, 39959, 42388
    ]

    historical_returns = get_historical_returns(start_year, 2020, simulation_years)
    
    balances = [base_balance]
    current_balance = base_balance
    withdrawals = [0]
    cumulative_withdrawals = [0]
    
    for year, return_rate in enumerate(historical_returns, 1):
        if current_balance > 0:
            current_balance *= (1 + return_rate)
        
        if year > 15 and year - 16 < len(withdrawal_schedule):
            withdrawal = withdrawal_schedule[year - 16]
            current_balance = max(0, current_balance - withdrawal)
            withdrawals.append(withdrawal)
            cumulative_withdrawals.append(cumulative_withdrawals[-1] + withdrawal)
        else:
            withdrawals.append(0)
            cumulative_withdrawals.append(cumulative_withdrawals[-1])
        
        balances.append(current_balance)
    
    return {
        'balances': balances,
        'withdrawals': withdrawals,
        'cumulative_withdrawals': cumulative_withdrawals,
        'returns': historical_returns
    }

# Run simulations for different start years
start_years = range(1924, 2021)  # From 1924 to 2020
intervals = [15, 20, 25, 30]  # Years after start
balance_at_intervals = {interval: [] for interval in intervals}
entry_years = []

# Create lists to store data for CSV
csv_data = []

for start_year in start_years:
    results = run_simulation(start_year, max(intervals))
    entry_years.append(start_year)
    
    # Prepare row for CSV
    row_data = {'Entry_Year': start_year}
    has_zero_balance = False
    
    for interval in intervals:
        if interval < len(results['balances']):
            balance = results['balances'][interval]
            balance_at_intervals[interval].append(balance)
            row_data[f'Balance_After_{interval}_Years'] = balance
            
            # Check for non-positive balance
            if balance <= 0:
                has_zero_balance = True
                print(f"WARNING: Non-positive balance (${balance:,.2f}) found for entry year {start_year} after {interval} years")
        else:
            balance_at_intervals[interval].append(None)
            row_data[f'Balance_After_{interval}_Years'] = None
    
    row_data['Has_Zero_Balance'] = has_zero_balance
    csv_data.append(row_data)

# Create DataFrame and save to CSV
df_results = pd.DataFrame(csv_data)
csv_filename = 'simulation_results.csv'
df_results.to_csv(csv_filename, index=False)
print(f"\nResults saved to {csv_filename}")

# Print summary of zero balances
zero_balance_count = df_results['Has_Zero_Balance'].sum()
if zero_balance_count > 0:
    print(f"\nFound {zero_balance_count} entry years with non-positive balances")
    print("\nEntry years with non-positive balances:")
    for _, row in df_results[df_results['Has_Zero_Balance']].iterrows():
        print(f"Entry Year: {int(row['Entry_Year'])}")
        for interval in intervals:
            balance = row[f'Balance_After_{interval}_Years']
            if balance is not None and balance <= 0:
                print(f"  After {interval} years: ${balance:,.2f}")

# Create the plot
plt.figure(figsize=(15, 10))

# Add horizontal balance reference lines
balance_levels = [100000, 200000, 300000, 400000, 500000]
for balance in balance_levels:
    plt.axhline(y=balance, color='gray', linestyle='--', alpha=0.5)

# Plot balance at each interval
markers = ['o', 's', '^', 'D']
colors = sns.color_palette("husl", len(intervals))

for interval, marker, color in zip(intervals, markers, colors):
    valid_data = [(year, bal) for year, bal in zip(entry_years, balance_at_intervals[interval]) if bal is not None]
    if valid_data:
        years, balances = zip(*valid_data)
        if interval == 15:
            label = f'Balance after {interval} years (before withdrawals)'
        else:
            label = f'Balance after {interval} years with withdrawals'
        plt.plot(years, balances, label=label, marker=marker, markersize=8, 
                linewidth=2, color=color, markeredgecolor='white', markeredgewidth=1)

plt.title('Investment Balance at Different Intervals by Entry Year\n(Starting with $100,000)', pad=20, fontsize=14)
plt.xlabel('Entry Year', fontsize=12)
plt.ylabel('Balance ($)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print("-" * 80)
for interval in intervals:
    valid_balances = [b for b in balance_at_intervals[interval] if b is not None]
    if valid_balances:
        print(f"\nInterval: {interval} years")
        print(f"Average Balance: ${np.mean(valid_balances):,.2f}")
        print(f"Minimum Balance: ${min(valid_balances):,.2f}")
        print(f"Maximum Balance: ${max(valid_balances):,.2f}")
        best_year_idx = balance_at_intervals[interval].index(max(valid_balances))
        worst_year_idx = balance_at_intervals[interval].index(min(valid_balances))
        print(f"Best Entry Year: {entry_years[best_year_idx]}")
        print(f"Worst Entry Year: {entry_years[worst_year_idx]}")
        print(f"Number of data points: {len(valid_balances)}") 