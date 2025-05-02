import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Callable

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)

def calculate_investment_trajectory(
    initial_balance: float,
    annual_returns: Union[float, List[float]],
    withdrawal_schedule: Union[float, List[float], Callable[[int, float], float]],
    withdrawal_start_year: int,
    max_years: int = 50,
    start_year: int = None
) -> dict:
    """
    Calculate investment balance and withdrawals over time.
    Balance can reach zero, but withdrawals continue as scheduled.
    
    Parameters:
    -----------
    initial_balance : float
        Starting amount in the investment account
    annual_returns : float or List[float]
        Annual return rates as decimal (e.g., 0.10 for 10%)
        Can be single float or list of returns for each year
    withdrawal_schedule : float or List[float] or Callable
        Can be one of:
        - float: Fixed annual withdrawal amount
        - List[float]: List of withdrawal amounts for each year
        - Callable: Function that takes (year, current_balance) and returns withdrawal amount
    withdrawal_start_year : int
        Year to start withdrawals (can be calendar year or relative year)
    max_years : int, optional
        Number of years to simulate (default 50)
    start_year : int, optional
        Calendar year to start from (if None, uses relative years)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'balances': array of account balances for each year
        - 'withdrawals': array of withdrawals for each year
        - 'cumulative_withdrawals': array of cumulative withdrawals for each year
        - 'years': array of years (calendar years if start_year provided, else relative years)
        - 'returns': array of return rates used for each year
    """
    # Initialize arrays to store results
    balances = np.zeros(max_years)
    withdrawals = np.zeros(max_years)
    cumulative_withdrawals = np.zeros(max_years)
    returns = np.zeros(max_years)
    
    # Create years array (either calendar or relative years)
    if start_year is not None:
        years = np.arange(start_year, start_year + max_years)
        withdrawal_start_year_relative = withdrawal_start_year - start_year
    else:
        years = np.arange(max_years)
        withdrawal_start_year_relative = withdrawal_start_year
    
    # Set initial balance
    balances[0] = initial_balance
    
    # Convert single return rate to list if needed
    if isinstance(annual_returns, (int, float)):
        returns[:] = annual_returns
    else:
        # Repeat the return rates pattern if list is shorter than max_years
        returns = np.array([annual_returns[i % len(annual_returns)] for i in range(max_years)])
    
    # Convert fixed withdrawal amount to a function
    if isinstance(withdrawal_schedule, (int, float)):
        fixed_amount = float(withdrawal_schedule)
        withdrawal_schedule = lambda year, balance: fixed_amount
    # Convert list of withdrawals to a function
    elif isinstance(withdrawal_schedule, list):
        withdrawal_list = withdrawal_schedule
        withdrawal_schedule = lambda year, balance: withdrawal_list[min(year - withdrawal_start_year_relative, len(withdrawal_list)-1)] if year >= withdrawal_start_year_relative else 0
    
    # Simulate account growth and withdrawals over time
    for year_idx in range(1, max_years):
        # Apply growth to previous year's balance (only if balance is positive)
        if balances[year_idx-1] > 0:
            balances[year_idx] = balances[year_idx-1] * (1 + returns[year_idx])
        
        # Apply withdrawal if we've reached the withdrawal start year
        if year_idx >= withdrawal_start_year_relative:
            # Calculate withdrawal amount based on the schedule
            withdrawal = withdrawal_schedule(year_idx, balances[year_idx])
            
            # Record the withdrawal
            withdrawals[year_idx] = withdrawal
            cumulative_withdrawals[year_idx] = cumulative_withdrawals[year_idx-1] + withdrawal
            
            # Update balance (can go to zero but not negative)
            balances[year_idx] = max(0, balances[year_idx] - withdrawal)
        else:
            cumulative_withdrawals[year_idx] = cumulative_withdrawals[year_idx-1]
            
    return {
        'balances': balances,
        'withdrawals': withdrawals,
        'cumulative_withdrawals': cumulative_withdrawals,
        'years': years,
        'returns': returns
    }

def plot_investment_trajectory(results: dict, title: str = "Investment Trajectory Over Time"):
    """
    Plot the investment balance and cumulative withdrawals over time.
    Uses seaborn for enhanced aesthetics.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for a single scenario
    title : str
        Title for the plot
    """
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Plot balance and withdrawals on primary axis
    line1 = sns.lineplot(x=results['years'], y=results['balances'],
                label='Account Balance',
                linewidth=2.5, marker='o', markersize=6,
                color='orange',
                markeredgecolor='white',
                markeredgewidth=1,
                ax=ax1)
    
    line2 = sns.lineplot(x=results['years'], y=results['cumulative_withdrawals'],
                label='Cumulative Withdrawals',
                linewidth=2.5, marker='s', markersize=6,
                color='gray', linestyle='--',
                markeredgecolor='white',
                markeredgewidth=1,
                ax=ax1)
    
    # Create secondary axis for return rates
    ax2 = ax1.twinx()
    
    # Plot light grey connecting lines for returns
    line3 = ax2.plot(results['years'], results['returns'] * 100,
                     color='lightgray', linewidth=1, zorder=4,
                     label='Annual Return Rate')[0]
    
    # Plot return rate points on top of the lines
    scatter = ax2.scatter(results['years'], results['returns'] * 100,  # Convert to percentage
               label='_nolegend_',  # Don't show points in legend
               s=80,  # Point size
               color='royalblue',
               edgecolor='white',
               linewidth=1,
               zorder=5)  # Ensure points are drawn on top
    
    # Calculate and plot average return line
    avg_return = np.mean(results['returns']) * 100
    avg_line = ax2.axhline(y=avg_return, color='navy', linestyle=':',
                label=f'Avg Return ({avg_return:.1f}%)', alpha=0.8)
    
    # Customize primary axis (left - dollar amounts)
    ax1.set_xlabel('Years', labelpad=10)
    ax1.set_ylabel('Amount ($)', labelpad=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Customize secondary axis (right - percentages)
    ax2.set_ylabel('Annual Return Rate (%)', labelpad=10, color='royalblue')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    ax2.tick_params(axis='y', colors='royalblue')
    
    # Set title
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    
    # Create combined legend
    all_lines = [line1.lines[0], line2.lines[0], line3, avg_line]
    all_labels = ['Account Balance', 'Cumulative Withdrawals', 
                 'Annual Return Rate', f'Avg Return ({avg_return:.1f}%)']
    ax1.legend(all_lines, all_labels,
              bbox_to_anchor=(1.15, 1), loc='upper left',
              frameon=True, fancybox=True, shadow=True)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Add background color
    ax1.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # User's specific example with variable returns and withdrawals
    base_balance = 100000
    bonus = 0
    initial_balance = base_balance + bonus
    
    # Annual returns as percentages
    annual_return_rates = [
        0.00, 0.0357, 0.0627, 0.0939, 0.0602, 0.2295, 0.0291, 0.0046, 0.0470, 0.1254,
        0.00, 0.0279, 0.0642, 0.1141, 0.0607, 0.2571, 0.0246, 0.0032, 0.0412, 0.1287,
        0.00, 0.0179, 0.0536, 0.1130, 0.0462, 0.2392, 0.0151, 0.0010, 0.0466, 0.1171,
        0.00, 0.0357, 0.0641, 0.1010, 0.0612, 0.2251, 0.0309, 0.0056, 0.0466, 0.1171,
        0.00, 0.0357, 0.0641, 0.1010, 0.0612, 0.2251
    ]
    
    # Withdrawal schedule
    withdrawal_schedule = [
        17995, 22986, 23600, 23682, 24757, 28307, 28307, 28889, 30709,
        34912, 36958, 49029, 50199, 50322, 52666, 58835, 58835, 60937,
        64845, 71392, 75758, 92809, 95681, 96212, 100693, 112488, 112488,
        116508, 123979, 136497, 144844
    ]
    
    year_start = 2025
    withdrawal_start_year = 2041
    withdrawal_end_year = 2071
    
    # Calculate max years needed
    max_years = withdrawal_end_year - year_start + 1
    
    print("\nRunning simulation with user's parameters:")
    print(f"Base Balance: ${base_balance:,}")
    print(f"Bonus: ${bonus:,}")
    print(f"Total Initial Balance: ${initial_balance:,}")
    print(f"Withdrawal Start Year: {withdrawal_start_year}")
    print(f"Withdrawal End Year: {withdrawal_end_year}")
    print(f"Number of years in simulation: {max_years}")
    
    # Calculate trajectory
    results = calculate_investment_trajectory(
        initial_balance=initial_balance,
        annual_returns=annual_return_rates,
        withdrawal_schedule=withdrawal_schedule,
        withdrawal_start_year=withdrawal_start_year,
        max_years=max_years,
        start_year=year_start
    )
    
    # Plot scenario
    plot_investment_trajectory(
        results,
        title=f"Investment Trajectory {year_start}-{withdrawal_end_year}\n(Variable Returns & Withdrawals)"
    )
    
    # Print key statistics
    final_balance = results['balances'][-1]
    total_withdrawn = results['cumulative_withdrawals'][-1]
    avg_return = np.mean(results['returns']) * 100
    
    print(f"\nKey Statistics:")
    print(f"Final Balance: ${final_balance:,.2f}")
    print(f"Total Withdrawn: ${total_withdrawn:,.2f}")
    print(f"Average Return Rate: {avg_return:.2f}%")
    
    # Find when balance hits zero (if it does)
    zero_balance_years = results['years'][results['balances'] == 0]
    if len(zero_balance_years) > 0:
        first_zero_year = zero_balance_years[0]
        print(f"Balance hits zero in: {first_zero_year}")
        remaining_withdrawals = total_withdrawn - results['cumulative_withdrawals'][results['years'] == first_zero_year][0]
        print(f"Total withdrawals after balance hits zero: ${remaining_withdrawals:,.2f}") 