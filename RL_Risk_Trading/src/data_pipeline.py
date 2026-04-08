import os
import yfinance as yf
import pandas as pd

def fetch_and_process_data():
    print("Fetching Nifty 50 and India VIX data from Yahoo Finance...")
    
    # 1. Download Data (From 2015 to April 2026)
    nifty = yf.download('^NSEI', start='2015-01-01', end='2026-04-01')
    vix = yf.download('^INDIAVIX', start='2015-01-01', end='2026-04-01')

    # We only need the 'Close' prices
    df = pd.DataFrame()
    df['Close'] = nifty['Close']
    df['VIX'] = vix['Close']

    # Drop any days where the market was closed or data is missing
    df = df.dropna()

    print("Calculating Technical Indicators (RSI & MACD)...")
    
    # 2. Calculate RSI (Relative Strength Index) - Momentum Indicator
    # Math: Compares the magnitude of recent gains to recent losses over 14 days.
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 3. Calculate MACD (Moving Average Convergence Divergence) - Trend Indicator
    # Math: Subtracts the 26-day exponential moving average (EMA) from the 12-day EMA.
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema_12 - ema_26

    # Drop the first few rows that have NaN values because of the rolling averages
    df = df.dropna().reset_index()

    # 4. Save the Final Dataset
    # Ensure the data folder exists
    os.makedirs('data/processed', exist_ok=True)
    save_path = 'data/processed/nifty50_with_risk.csv'
    df.to_csv(save_path, index=False)
    
    print(f"Success! Dataset saved to {save_path}")
    print(f"Total trading days collected: {len(df)}")
    return df

# This ensures the code only runs if we execute this specific file
if __name__ == "__main__":
    fetch_and_process_data()