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
    annual_return: float,
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
    annual_return : float
        Annual return rate as a decimal (e.g., 0.10 for 10%)
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
    """
    # Initialize arrays to store results
    balances = np.zeros(max_years)
    withdrawals = np.zeros(max_years)
    cumulative_withdrawals = np.zeros(max_years)
    
    # Create years array (either calendar or relative years)
    if start_year is not None:
        years = np.arange(start_year, start_year + max_years)
        withdrawal_start_year_relative = withdrawal_start_year - start_year
    else:
        years = np.arange(max_years)
        withdrawal_start_year_relative = withdrawal_start_year
    
    # Set initial balance
    balances[0] = initial_balance
    
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
            balances[year_idx] = balances[year_idx-1] * (1 + annual_return)
        
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
        'years': years
    }

def plot_investment_trajectory(results_dict: dict, title: str = "Investment Trajectory Over Time"):
    """
    Plot the investment balance for multiple return rates and cumulative withdrawals over time.
    Uses seaborn for enhanced aesthetics.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results for each return rate
    title : str
        Title for the plot
    """
    plt.figure(figsize=(15, 8))
    
    # Create custom color palette for balance lines
    n_returns = len(results_dict)
    balance_colors = sns.color_palette("husl", n_returns)
    
    # Plot balance for each return rate
    for (return_rate, results), color in zip(results_dict.items(), balance_colors):
        sns.lineplot(x=results['years'], y=results['balances'],
                    label=f'Balance ({return_rate}% return)',
                    linewidth=2.5, marker='o', markersize=6,
                    color=color, markeredgecolor='white',
                    markeredgewidth=1)
    
    # Plot cumulative withdrawals (same for all scenarios)
    sns.lineplot(x=results_dict[list(results_dict.keys())[0]]['years'],
                y=results_dict[list(results_dict.keys())[0]]['cumulative_withdrawals'],
                label='Cumulative Withdrawals',
                linewidth=2.5, marker='s', markersize=6,
                color='gray', linestyle='--',
                markeredgecolor='white',
                markeredgewidth=1)
    
    # Customize the plot
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Years', labelpad=10)
    plt.ylabel('Amount ($)', labelpad=10)
    
    # Format y-axis with dollar amounts
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
              frameon=True, fancybox=True, shadow=True)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Add background color
    plt.gca().set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # User's specific example
    base_balance = 100000
    bonus = 0
    initial_balance = base_balance + bonus
    annual_returns = [0.04, 0.06, 0.07, 0.08]  # 4% to 8%
    year_start = 2025
    withdrawal_start_year = 2041
    withdrawal_end_year = 2072
    
    withdrawal_schedule = [
        12282, 13024, 13284, 14087, 14368, 15236, 15541, 16479, 16809, 17824, 18181,
        19278, 19664, 20852, 21269, 22562, 23013, 24412, 24900, 26414, 26943, 28581,
        29152, 30925, 31543, 33461, 34130, 36206, 36930, 39175, 39959, 42388
    ]
    
    # Calculate max years needed
    max_years = withdrawal_end_year - year_start + 1
    
    print("\nRunning simulation with user's parameters:")
    print(f"Base Balance: ${base_balance:,}")
    print(f"Bonus: ${bonus:,}")
    print(f"Total Initial Balance: ${initial_balance:,}")
    print(f"Annual Returns: {', '.join(f'{r*100}%' for r in annual_returns)}")
    print(f"Withdrawal Start Year: {withdrawal_start_year}")
    print(f"Withdrawal End Year: {withdrawal_end_year}")
    print(f"Number of years in simulation: {max_years}")
    
    # Calculate trajectories for each return rate
    results_dict = {}
    for annual_return in annual_returns:
        results = calculate_investment_trajectory(
            initial_balance=initial_balance,
            annual_return=annual_return,
            withdrawal_schedule=withdrawal_schedule,
            withdrawal_start_year=withdrawal_start_year,
            max_years=max_years,
            start_year=year_start
        )
        results_dict[f"{annual_return*100:.0f}"] = results
    
    # Plot all scenarios
    plot_investment_trajectory(
        results_dict,
        title=f"Investment Trajectory Comparison {year_start}-{withdrawal_end_year}"
    )
    
    # Print key statistics for each return rate
    print(f"\nKey Statistics:")
    for return_rate, results in results_dict.items():
        final_balance = results['balances'][-1]
        total_withdrawn = results['cumulative_withdrawals'][-1]
        print(f"\nReturn Rate: {return_rate}%")
        print(f"Final Balance: ${final_balance:,.2f}")
        print(f"Total Withdrawn: ${total_withdrawn:,.2f}")
        
        # Find when balance hits zero (if it does)
        zero_balance_years = results['years'][results['balances'] == 0]
        if len(zero_balance_years) > 0:
            first_zero_year = zero_balance_years[0]
            print(f"Balance hits zero in: {first_zero_year}")
            remaining_withdrawals = total_withdrawn - results['cumulative_withdrawals'][results['years'] == first_zero_year][0]
            print(f"Total withdrawals after balance hits zero: ${remaining_withdrawals:,.2f}") 