import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO
from environment import RiskAwareTradingEnv

def run_simulation(model_path, env_mode, df):
    """Runs the simulation and returns Net Worth and Drawdown histories."""
    env = RiskAwareTradingEnv(df, mode=env_mode)
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    done = False
    net_worths = []
    drawdowns = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        if 'net_worth' in info:
            net_worths.append(info['net_worth'])
            drawdowns.append(info['drawdown'] * 100) # Convert to percentage
            
    return net_worths, drawdowns

def evaluate_blind_test():
    print("Loading Nifty 50 Data for BLIND TEST (2024-2026)...")
    df = pd.read_csv('data/processed/nifty50_with_risk.csv')

    # --- THE BLIND TEST SPLIT ---
    df['Date'] = pd.to_datetime(df['Date'])
    test_df = df[df['Date'] >= '2024-01-01'].reset_index(drop=True)

    print("Running Baseline (Buy & Hold Nifty 50)...")
    start_price = test_df.loc[0, 'Close']
    initial_shares = 100000 // start_price
    leftover_cash = 100000 % start_price
    
    bh_net_worths = [(leftover_cash + (initial_shares * price)) for price in test_df['Close']]
    bh_net_worths = bh_net_worths[1:] 
    dates = test_df['Date'][1:].values 

    print("Running Simulation: Vanilla AI (The Villain)...")
    vanilla_nw, vanilla_dd = run_simulation("models/ppo_vanilla.zip", 'vanilla', test_df)

    print("Running Simulation: Risk-Aware AI (The Hero)...")
    smart_nw, smart_dd = run_simulation("models/ppo_risk_aware.zip", 'risk_aware', test_df)

    # --- METRICS ALIGNMENT ---
    min_len = min(len(bh_net_worths), len(vanilla_nw), len(smart_nw))
    dates = dates[:min_len]
    bh_nw = bh_net_worths[:min_len]
    vanilla_nw = vanilla_nw[:min_len]
    smart_nw = smart_nw[:min_len]
    vanilla_dd = vanilla_dd[:min_len]
    smart_dd = smart_dd[:min_len]

    bh_final = bh_nw[-1]
    vanilla_final = vanilla_nw[-1]
    smart_final = smart_nw[-1]
    
    bh_ret = ((bh_final-100000)/100000)*100
    vanilla_ret = ((vanilla_final-100000)/100000)*100
    smart_ret = ((smart_final-100000)/100000)*100

    vanilla_max_dd = max(vanilla_dd)
    smart_max_dd = max(smart_dd)

    # --- SAVE METRICS TO TEXT FILE (For Member 2) ---
    os.makedirs('results', exist_ok=True)
    metrics_text = f"""FINAL BLIND TEST METRICS (2024 - 2026)
----------------------------------------
1. Buy & Hold Market  : Rs {bh_final:,.2f} ({bh_ret:.2f}%)
2. Vanilla AI (Greedy): Rs {vanilla_final:,.2f} ({vanilla_ret:.2f}%) | Max Drawdown: {vanilla_max_dd:.2f}%
3. Risk-Aware AI      : Rs {smart_final:,.2f} ({smart_ret:.2f}%) | Max Drawdown: {smart_max_dd:.2f}%
"""
    print("\n" + metrics_text)
    with open('results/metrics_table.txt', 'w') as f:
        f.write(metrics_text)

    # --- GENERATE FIGURES FOR THE PAPER ---
    print("Generating High-Resolution Figures for the Research Paper...")
    
    # FIGURE 1 & 3: Portfolio Growth (Blind Test)
    plt.figure(figsize=(12, 6))
    plt.plot(dates, bh_nw, label='Nifty 50 (Buy & Hold)', color='gray', linestyle='dashed', alpha=0.7)
    plt.plot(dates, vanilla_nw, label='Vanilla PPO', color='red', alpha=0.8)
    plt.plot(dates, smart_nw, label='Risk-Aware PPO (Ours)', color='green', linewidth=2)
    plt.title('Figure 1: Blind Test Portfolio Growth (2024 - 2026)', fontsize=14, fontweight='bold')
    plt.ylabel('Portfolio Value (Rs)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figure_1_portfolio_growth.png', dpi=300)

    # FIGURE 2: Max Drawdown Comparison
    plt.figure(figsize=(12, 5))
    plt.plot(dates, vanilla_dd, label='Vanilla AI Drawdown %', color='red', alpha=0.6)
    plt.plot(dates, smart_dd, label='Risk-Aware AI Drawdown %', color='green', linewidth=2)
    plt.fill_between(dates, vanilla_dd, color='red', alpha=0.2)
    plt.fill_between(dates, smart_dd, color='green', alpha=0.2)
    plt.title('Figure 2: Risk Analysis - Maximum Drawdown %', fontsize=14, fontweight='bold')
    plt.ylabel('Drawdown (Percentage Drop)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figure_2_max_drawdown.png', dpi=300)

    print("Success! All files saved to the 'results/' folder.")

if __name__ == "__main__":
    evaluate_blind_test()