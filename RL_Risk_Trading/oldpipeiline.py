import yfinance as yf
import pandas as pd

data = yf.download("AAPL",start="2020-01-01",end="2020-12-31")


print(type(data))
print(data.columns)
print(data.head())


print(type(data["Close"]))
print(type(data[("Close", "AAPL")]))

df=pd.DataFrame(data["Close","AAPL"])
df.columns = df.columns.get_level_values(0)
print(df.head())
print(df.columns)