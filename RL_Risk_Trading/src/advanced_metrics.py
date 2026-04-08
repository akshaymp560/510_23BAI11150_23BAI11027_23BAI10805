import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from environment import RiskAwareTradingEnv

def calculate_metrics(net_worths):
    """Calculates advanced financial metrics from an array of daily net worths."""
    # Convert net worths to a pandas Series for easy math
    nw = pd.Series(net_worths)
    
    # Calculate daily percentage returns
    daily_returns = nw.pct_change().dropna()
    
    # 1. Total Return
    total_return = ((nw.iloc[-1] - nw.iloc[0]) / nw.iloc[0]) * 100
    
    # 2. Annualized Return (Assuming 252 trading days in a year)
    days = len(nw)
    annualized_return = (((nw.iloc[-1] / nw.iloc[0]) ** (252 / days)) - 1) * 100
    
    # 3. Sharpe Ratio (Risk-Adjusted Return. >1 is good, >2 is amazing)
    # Assuming a 5% annual risk-free rate (0.05 / 252 daily)
    risk_free_daily = 0.05 / 252
    excess_returns = daily_returns - risk_free_daily
    if excess_returns.std() == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
    # 4. Win Rate (Equivalent to "Accuracy" - Percentage of days that made money)
    win_rate = (len(daily_returns[daily_returns > 0]) / len(daily_returns)) * 100
    
    return total_return, annualized_return, sharpe_ratio, win_rate

def run_simulation(model_path, env_mode, df):
    """Runs the simulation and returns the net worth history."""
    env = RiskAwareTradingEnv(df, mode=env_mode)
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    done = False
    net_worths = [100000] # Starting balance
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        if 'net_worth' in info:
            net_worths.append(info['net_worth'])
            
    return net_worths

def generate_report():
    print("Loading Data for Advanced Metrics Test...")
    df = pd.read_csv('data/processed/nifty50_with_risk.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    test_df = df[df['Date'] >= '2024-01-01'].reset_index(drop=True)

    print("Running Simulations...")
    
    # 1. Buy & Hold Baseline
    start_price = test_df.loc[0, 'Close']
    initial_shares = 100000 // start_price
    leftover_cash = 100000 % start_price
    bh_net_worths = [100000] + [(leftover_cash + (initial_shares * price)) for price in test_df['Close'][1:]]
    
    # 2. Vanilla AI
    vanilla_nw = run_simulation("models/ppo_vanilla.zip", 'vanilla', test_df)
    
    # 3. Risk-Aware AI
    smart_nw = run_simulation("models/ppo_risk_aware.zip", 'risk_aware', test_df)

    # Calculate Metrics
    bh_metrics = calculate_metrics(bh_net_worths)
    vanilla_metrics = calculate_metrics(vanilla_nw)
    smart_metrics = calculate_metrics(smart_nw)

    # Print the Report
    report = f"""
=========================================================
📊 ADVANCED AI PERFORMANCE REPORT (Blind Test 2024-2026)
=========================================================

1. BUY & HOLD MARKET BASELINE
--------------------------------
Total Return      : {bh_metrics[0]:.2f}%
Annualized Return : {bh_metrics[1]:.2f}%
Sharpe Ratio      : {bh_metrics[2]:.2f}
Win Rate          : {bh_metrics[3]:.2f}% (Days in profit)

2. VANILLA PPO (Greedy AI)
--------------------------------
Total Return      : {vanilla_metrics[0]:.2f}%
Annualized Return : {vanilla_metrics[1]:.2f}%
Sharpe Ratio      : {vanilla_metrics[2]:.2f}
Win Rate          : {vanilla_metrics[3]:.2f}% (Days in profit)

3. RISK-AWARE PPO (Our AI)
--------------------------------
Total Return      : {smart_metrics[0]:.2f}%
Annualized Return : {smart_metrics[1]:.2f}%
Sharpe Ratio      : {smart_metrics[2]:.2f}
Win Rate          : {smart_metrics[3]:.2f}% (Days in profit)

=========================================================
"""
    print(report)
    
    # Save to file
    os.makedirs('results', exist_ok=True)
    with open('results/advanced_metrics_report.txt', 'w',encoding='utf-8') as f:
        f.write(report)
    print("✅ Advanced report saved to results/advanced_metrics_report.txt")

if __name__ == "__main__":
    generate_report()