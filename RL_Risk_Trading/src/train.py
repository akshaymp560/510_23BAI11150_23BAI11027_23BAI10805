import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import our custom environment from the local file
from environment import RiskAwareTradingEnv

def train_models():
    print("Loading the Nifty 50 World Map...")
    df = pd.read_csv('data/processed/nifty50_with_risk.csv')
    
    # --- PREVENTING DATA LEAKAGE (The Blind Test Fix) ---
    # We force the AI to ONLY learn from 2015 to 2023.
    # It will have zero knowledge of the 2024-2026 data!
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] < '2024-01-01'].reset_index(drop=True)
    # ----------------------------------------------------

    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)

    # ==========================================
    # 1. TRAIN THE "VANILLA" CONTROL GROUP
    # ==========================================
    print("\n" + "="*45)
    print("🧠 TRAINING BRAIN 1: THE VANILLA AI (Standard)")
    print("="*45)
    print("This AI ignores the VIX. It only cares about profit.")
    print(f"Training on historical data ({len(df)} days)... (This will take 1-3 minutes)")
    
    # Notice mode='vanilla'. This turns OFF your custom VIX penalty.
    env_vanilla = DummyVecEnv([lambda: RiskAwareTradingEnv(df, mode='vanilla')])
    
    # Initialize and train
    model_vanilla = PPO("MlpPolicy", env_vanilla, verbose=1, learning_rate=0.0005, ent_coef=0.01)
    model_vanilla.learn(total_timesteps=150000)
    
    # Save it
    vanilla_path = "models/ppo_vanilla"
    model_vanilla.save(vanilla_path)
    print(f"\n✅ Vanilla Brain saved to {vanilla_path}.zip")

    # ==========================================
    # 2. TRAIN THE "RISK-AWARE" EXPERIMENTAL GROUP
    # ==========================================
    print("\n" + "="*45)
    print("🧠 TRAINING BRAIN 2: THE RISK-AWARE AI (Smart)")
    print("="*45)
    print("This AI uses your novel Reward Shaping & Panic Eject Button.")
    print(f"Training on historical data ({len(df)} days)... (This will take 1-3 minutes)")
    
    # Notice mode='risk_aware'. This turns ON your custom math.
    env_smart = DummyVecEnv([lambda: RiskAwareTradingEnv(df, mode='risk_aware')])
    
    # Initialize and train
    model_smart = PPO("MlpPolicy", env_smart, verbose=1, learning_rate=0.0005, ent_coef=0.01)
    model_smart.learn(total_timesteps=150000)
    
    # Save it
    smart_path = "models/ppo_risk_aware"
    model_smart.save(smart_path)
    print(f"\n✅ Risk-Aware Brain saved to {smart_path}.zip")

    print("\n🎉 Both models trained successfully! Ready for the Blind Test Showdown.")

if __name__ == "__main__":
    train_models()