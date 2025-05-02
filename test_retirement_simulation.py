import unittest
import pandas as pd
import numpy as np
from historical_sp500_simulation_refactored import (
    SimulationConfig, HistoricalReturns, SimulationResult, RetirementSimulator
)

class TestSimulationConfig(unittest.TestCase):
    def test_default_values(self):
        config = SimulationConfig()
        self.assertEqual(config.base_balance, 100000)
        self.assertEqual(config.default_return, 0.10)
        self.assertEqual(config.waiting_years, 15)
        self.assertEqual(len(config.withdrawal_schedule), 32)  # Should have 32 years of withdrawals
        
    def test_custom_values(self):
        custom_withdrawals = [1000, 2000, 3000]
        config = SimulationConfig(
            base_balance=200000,
            default_return=0.05,
            waiting_years=10,
            withdrawal_schedule=custom_withdrawals
        )
        self.assertEqual(config.base_balance, 200000)
        self.assertEqual(config.default_return, 0.05)
        self.assertEqual(config.waiting_years, 10)
        self.assertEqual(config.withdrawal_schedule, custom_withdrawals)

class TestHistoricalReturns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a mock CSV file for testing
        cls.test_data = pd.DataFrame({
            'Year': [2020, 2021, 2022],
            'Total_Return': [15.0, 20.0, -10.0]
        })
        cls.test_data.to_csv('test_returns.csv', index=False)
        
    def setUp(self):
        self.returns = HistoricalReturns('test_returns.csv')
    
    def test_data_loading(self):
        self.assertTrue('Annual_Return' in self.returns.df.columns)
        self.assertEqual(len(self.returns.df), 3)
    
    def test_return_conversion(self):
        # Test that Total_Return is properly converted to decimal
        self.assertAlmostEqual(self.returns.df.loc[0, 'Annual_Return'], 0.15)
        self.assertAlmostEqual(self.returns.df.loc[1, 'Annual_Return'], 0.20)
        self.assertAlmostEqual(self.returns.df.loc[2, 'Annual_Return'], -0.10)
    
    def test_get_return(self):
        # Test existing year
        self.assertAlmostEqual(self.returns.get_return(2020, 0.10), 0.15)
        # Test non-existing year
        self.assertAlmostEqual(self.returns.get_return(2023, 0.10), 0.10)

class TestSimulationResult(unittest.TestCase):
    def test_has_zero_balance(self):
        # Test with all positive balances
        result = SimulationResult(
            entry_year=2020,
            balances=[100000, 110000, 120000],
            withdrawals=[0, 1000, 1000],
            cumulative_withdrawals=[0, 1000, 2000],
            returns=[0.1, 0.1]
        )
        self.assertFalse(result.has_zero_balance())
        
        # Test with zero balance
        result = SimulationResult(
            entry_year=2020,
            balances=[100000, 0, 0],
            withdrawals=[0, 100000, 0],
            cumulative_withdrawals=[0, 100000, 100000],
            returns=[0.1, 0.1]
        )
        self.assertTrue(result.has_zero_balance())

class TestRetirementSimulator(unittest.TestCase):
    def setUp(self):
        # Create a simple configuration for testing
        self.config = SimulationConfig(
            base_balance=100000,
            default_return=0.10,
            waiting_years=2,  # Shorter waiting period for testing
            withdrawal_schedule=[10000, 10000]  # Simple withdrawal schedule
        )
        self.simulator = RetirementSimulator(self.config)
    
    def test_simulate_single_path(self):
        result = self.simulator.simulate_single_path(2020)
        
        # Test basic properties
        self.assertEqual(result.entry_year, 2020)
        self.assertEqual(len(result.returns), 4)  # 2 waiting years + 2 withdrawal years
        self.assertEqual(len(result.balances), 5)  # Initial balance + 4 years
        self.assertEqual(len(result.withdrawals), 5)  # Initial 0 + 4 years
        
        # Test initial conditions
        self.assertEqual(result.balances[0], 100000)
        self.assertEqual(result.withdrawals[0], 0)
        self.assertEqual(result.cumulative_withdrawals[0], 0)
        
        # Test that withdrawals start after waiting period
        self.assertEqual(result.withdrawals[1], 0)
        self.assertEqual(result.withdrawals[2], 0)
        self.assertEqual(result.withdrawals[3], 10000)
        self.assertEqual(result.withdrawals[4], 10000)

    def test_balance_calculation(self):
        """Test the actual balance calculations with known returns"""
        # Create a test configuration with known values
        test_config = SimulationConfig(
            base_balance=100000,
            default_return=0.10,  # 10% return
            waiting_years=1,
            withdrawal_schedule=[20000]  # One withdrawal
        )
        simulator = RetirementSimulator(test_config)
        
        # Mock the historical returns to always return 10%
        simulator.historical_returns.get_return = lambda year, default: 0.10
        
        result = simulator.simulate_single_path(2020)
        
        # Manual calculation verification:
        # Year 0: Initial balance = 100000
        self.assertEqual(result.balances[0], 100000)
        
        # Year 1: After 10% return = 100000 * (1 + 0.10) = 110000, no withdrawal
        self.assertEqual(result.balances[1], 110000)
        
        # Year 2: After 10% return = 110000 * (1 + 0.10) = 121000, then withdraw 20000
        expected_final_balance = 121000 - 20000
        self.assertEqual(result.balances[2], expected_final_balance)

    def test_zero_balance_scenario(self):
        """Test scenario where balance goes to zero due to large withdrawals"""
        test_config = SimulationConfig(
            base_balance=100000,
            default_return=0.10,
            waiting_years=1,
            withdrawal_schedule=[150000]  # Withdrawal larger than balance
        )
        simulator = RetirementSimulator(test_config)
        
        # Mock the historical returns to always return 10%
        simulator.historical_returns.get_return = lambda year, default: 0.10
        
        result = simulator.simulate_single_path(2020)
        
        # Calculate expected values:
        # Year 0: 100000
        # Year 1: 100000 * 1.1 = 110000 (no withdrawal)
        # Year 2: 110000 * 1.1 = 121000, then withdraw 150000 -> should go to 0
        
        expected_balances = [100000, 110000, 0]
        self.assertEqual(result.balances, expected_balances)
        
        # Verify that the withdrawal was limited to the available balance
        self.assertEqual(result.withdrawals[2], 121000)  # Should only withdraw what's available
        self.assertTrue(result.has_zero_balance())

    def test_cumulative_withdrawals(self):
        """Test that cumulative withdrawals are calculated correctly"""
        test_config = SimulationConfig(
            base_balance=100000,
            default_return=0.10,
            waiting_years=1,
            withdrawal_schedule=[10000, 20000, 30000]
        )
        simulator = RetirementSimulator(test_config)
        
        result = simulator.simulate_single_path(2020)
        
        # Verify cumulative withdrawals
        expected_cumulative = [0, 0, 10000, 30000, 60000]
        for actual, expected in zip(result.cumulative_withdrawals, expected_cumulative):
            self.assertEqual(actual, expected)

def main():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main() 