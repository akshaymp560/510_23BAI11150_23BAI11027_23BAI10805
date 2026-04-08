import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class RiskAwareTradingEnv(gym.Env):
    """
    A custom trading environment for OpenAI Gymnasium.
    Upgraded with Percentage-Based Rewards and 'Reward Shaping' 
    to perfectly balance risk-aversion and profit-seeking.
    """
    def __init__(self, df, initial_balance=100000, mode='risk_aware'):
        super(RiskAwareTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # 'mode' switch so we can train a 'vanilla' baseline 
        self.mode = mode 
        
        self.features = ['Close', 'VIX', 'RSI_14', 'MACD_12_26_9']
        self.action_space = spaces.Discrete(3) # 0: Sell, 1: Hold, 2: Buy
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.df.loc[self.current_step, self.features].values
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        terminated = self.current_step >= len(self.df) - 1
        if terminated:
            return self._get_obs(), 0, terminated, False, {}

        current_price = self.df.loc[self.current_step, 'Close']
        current_vix = self.df.loc[self.current_step, 'VIX']
        
        prev_net_worth = self.net_worth

        # --- 1. EXECUTE TRADE ---
        if action == 0 and self.shares_held > 0: # SELL
            self.balance += self.shares_held * current_price
            self.shares_held = 0
        elif action == 2 and self.balance > current_price: # BUY
            shares_bought = self.balance // current_price
            self.balance -= shares_bought * current_price
            self.shares_held += shares_bought

        # Update Net Worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # --- 2. CALCULATE PROFIT (PERCENTAGES) ---
        # Neural Networks hate large raw numbers. We convert profit to a daily percentage!
        # The max(..., 1) is a safeguard to prevent division by zero errors.
        step_profit_pct = ((self.net_worth - prev_net_worth) / max(prev_net_worth, 1)) * 100
        drawdown = (self.max_net_worth - self.net_worth) / max(self.max_net_worth, 1)
        
        # --- 3. ACADEMIC NOVELTY: Reward Shaping ---
        risk_penalty = 0
        
        if self.mode == 'risk_aware':
            # RULE A: The Panic Eject Button
            # If the market is crashing (VIX >= 22) and it holds shares, punish it.
            if current_vix >= 22.0 and self.shares_held > 0:
                risk_penalty = 0.5  # Reduced from 5.0 so it isn't traumatized
                
            # RULE B: The Inflation Nudge (The Couch Potato Fix)
            # If the market is safe (VIX < 20) but it's hiding in cash, punish it!
            elif current_vix < 20.0 and self.shares_held == 0:
                risk_penalty = 0.1  # A tiny poke to force it to buy stocks
                
        # --- 4. FINAL REWARD EQUATION ---
        reward = step_profit_pct - risk_penalty

        info = {
            'net_worth': self.net_worth,
            'drawdown': drawdown,
            'reward': reward,
            'vix': current_vix
        }

        # OpenAI Gym expects the reward to be a float
        return self._get_obs(), float(reward), terminated, False, info

# ==========================================
# TEST BLOCK: Ensure the environment works
# ==========================================
if __name__ == "__main__":
    print("Testing the Upgraded Environment...")
    df = pd.read_csv('data/processed/nifty50_with_risk.csv')
    
    env = RiskAwareTradingEnv(df, mode='risk_aware')
    obs, _ = env.reset()
    
    print("\nTaking 5 random actions to test the new Physics engine:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        action_name = ["SELL", "HOLD", "BUY"][action]
        
        print(f"Day {i+1} | Action: {action_name:<4} | VIX: {info['vix']:.2f} | Reward (Pct): {reward:.4f}%")
    
    print("\nEnvironment upgrade successful! Ready for Ablation Study.")