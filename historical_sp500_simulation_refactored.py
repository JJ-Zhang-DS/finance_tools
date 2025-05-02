import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Optional

# Configuration
@dataclass
class SimulationConfig:
    base_balance: float = 100000
    default_return: float = 0.10
    waiting_years: int = 15  # Years before withdrawals start
    withdrawal_schedule: List[float] = None
    
    def __post_init__(self):
        if self.withdrawal_schedule is None:
            self.withdrawal_schedule = [
                12282, 13024, 13284, 14087, 14368, 15236, 15541, 16479, 16809, 17824,
                18181, 19278, 19664, 20852, 21269, 22562, 23013, 24412, 24900, 26414,
                26943, 28581, 29152, 30925, 31543, 33461, 34130, 36206, 36930, 39175,
                39959, 42388
            ]

class HistoricalReturns:
    def __init__(self, data_path: str = 'data/spy500_history.csv'):
        self.df = pd.read_csv(data_path)
        self.df['Annual_Return'] = self.df['Total_Return'] / 100  # Convert to decimal
        
    def get_return(self, year: int, default_return: float) -> float:
        """Get the return for a specific year, using default if not available"""
        year_data = self.df[self.df['Year'] == year]
        if not year_data.empty and not pd.isna(year_data['Annual_Return'].iloc[0]):
            return year_data['Annual_Return'].iloc[0]
        return default_return

@dataclass
class SimulationResult:
    entry_year: int
    balances: List[float]
    withdrawals: List[float]
    cumulative_withdrawals: List[float]
    returns: List[float]
    
    def has_zero_balance(self) -> bool:
        return any(balance <= 0 for balance in self.balances)

class RetirementSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.historical_returns = HistoricalReturns()
        
    def simulate_single_path(self, entry_year: int) -> SimulationResult:
        """Run simulation for a single entry year"""
        total_years = self.config.waiting_years + len(self.config.withdrawal_schedule)
        returns = []
        balances = [self.config.base_balance]
        withdrawals = [0]
        cumulative_withdrawals = [0]
        current_balance = self.config.base_balance
        
        # Calculate returns for each year
        for year_offset in range(total_years):
            target_year = entry_year + year_offset
            annual_return = self.historical_returns.get_return(
                target_year, self.config.default_return)
            returns.append(annual_return)
            
            # Apply return (round to 2 decimal places to avoid floating point issues)
            if current_balance > 0:
                current_balance = round(current_balance * (1 + annual_return), 2)
            
            # Apply withdrawal if after waiting period
            if year_offset >= self.config.waiting_years:
                withdrawal_idx = year_offset - self.config.waiting_years
                if withdrawal_idx < len(self.config.withdrawal_schedule):
                    withdrawal = self.config.withdrawal_schedule[withdrawal_idx]
                    # If withdrawal would exceed balance, set balance to zero
                    if withdrawal >= current_balance:
                        withdrawal = current_balance
                        current_balance = 0
                    else:
                        current_balance = round(current_balance - withdrawal, 2)
                    withdrawals.append(withdrawal)
                    cumulative_withdrawals.append(round(cumulative_withdrawals[-1] + withdrawal, 2))
                else:
                    withdrawals.append(0)
                    cumulative_withdrawals.append(cumulative_withdrawals[-1])
            else:
                withdrawals.append(0)
                cumulative_withdrawals.append(cumulative_withdrawals[-1])
            
            balances.append(current_balance)
        
        return SimulationResult(
            entry_year=entry_year,
            balances=balances,
            withdrawals=withdrawals,
            cumulative_withdrawals=cumulative_withdrawals,
            returns=returns
        )

class SimulationAnalyzer:
    def __init__(self, results: List[SimulationResult], intervals: List[int]):
        self.results = results
        self.intervals = intervals
        
    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with summary statistics for each entry year"""
        data = []
        for result in self.results:
            row = {'Entry_Year': result.entry_year}
            for interval in self.intervals:
                if interval < len(result.balances):
                    row[f'Balance_After_{interval}_Years'] = result.balances[interval]
            row['Has_Zero_Balance'] = result.has_zero_balance()
            data.append(row)
        return pd.DataFrame(data)
    
    def plot_results(self):
        """Create visualization of simulation results"""
        plt.figure(figsize=(15, 10))
        
        # Add reference lines
        for balance in [100000, 200000, 300000, 400000, 500000]:
            plt.axhline(y=balance, color='gray', linestyle='--', alpha=0.5)
        
        # Plot balance at each interval
        markers = ['o', 's', '^', 'D']
        colors = sns.color_palette("husl", len(self.intervals))
        
        for interval, marker, color in zip(self.intervals, markers, colors):
            data = [(r.entry_year, r.balances[interval]) 
                   for r in self.results 
                   if interval < len(r.balances)]
            
            if data:
                years, balances = zip(*data)
                label = (f'Balance after {interval} years '
                        f'{"(before withdrawals)" if interval <= 15 else "with withdrawals"}')
                plt.plot(years, balances, label=label, marker=marker, markersize=8,
                        linewidth=2, color=color, markeredgecolor='white', markeredgewidth=1)
        
        plt.title('Investment Balance at Different Intervals by Entry Year\n(Starting with $100,000)',
                 pad=20, fontsize=14)
        plt.xlabel('Entry Year', fontsize=12)
        plt.ylabel('Balance ($)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    # Initialize configuration
    config = SimulationConfig()
    simulator = RetirementSimulator(config)
    
    # Run simulations
    results = []
    for entry_year in range(1924, 2021):
        result = simulator.simulate_single_path(entry_year)
        results.append(result)
    
    # Analyze results
    intervals = [15, 20, 25, 30]
    analyzer = SimulationAnalyzer(results, intervals)
    
    # Generate and save summary
    df_results = analyzer.create_summary_dataframe()
    df_results.to_csv('simulation_results.csv', index=False)
    print(f"\nResults saved to simulation_results.csv")
    
    # Print summary of zero balances
    zero_balance_results = [r for r in results if r.has_zero_balance()]
    if zero_balance_results:
        print(f"\nFound {len(zero_balance_results)} entry years with non-positive balances")
        print("\nEntry years with non-positive balances:")
        for result in zero_balance_results:
            print(f"Entry Year: {result.entry_year}")
            for interval in intervals:
                if interval < len(result.balances) and result.balances[interval] <= 0:
                    print(f"  After {interval} years: ${result.balances[interval]:,.2f}")
    
    # Create visualization
    analyzer.plot_results()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    for interval in intervals:
        balances = [r.balances[interval] for r in results if interval < len(r.balances)]
        if balances:
            print(f"\nInterval: {interval} years")
            print(f"Average Balance: ${np.mean(balances):,.2f}")
            print(f"Minimum Balance: ${min(balances):,.2f}")
            print(f"Maximum Balance: ${max(balances):,.2f}")
            best_result = max(results, key=lambda r: r.balances[interval] if interval < len(r.balances) else -float('inf'))
            worst_result = min(results, key=lambda r: r.balances[interval] if interval < len(r.balances) else float('inf'))
            print(f"Best Entry Year: {best_result.entry_year}")
            print(f"Worst Entry Year: {worst_result.entry_year}")
            print(f"Number of data points: {len(balances)}")

if __name__ == "__main__":
    main() 