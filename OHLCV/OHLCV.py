import numpy as np 
import pandas as pd 
import yfinance as yf

tech_tickers = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "IBM", "AMD", "MU", "INTC", "QCOM", "TXN", "ADI", "KLAC", "LRCX", "AMAT", "CRM", "ADBE",
    "NOW", "INTU", "ADSK",  "CDNS",  "SNPS", "PANW", "FTNT", "CRWD", "ZS", "CSCO", "ANET", "APH", "V", "MA", "PYPL", "PLTR", "APP",
    "ACN", "MSI", "GLW", "NXPI", "MPWR", 'CAT']

data = yf.download(tech_tickers, period="5y", auto_adjust=True)
data.columns = ['_'.join(col).strip() for col in data.columns.values]
data.ffill(inplace=True)  
data.bfill(inplace=True)

tickers = tech_tickers
dfs = []

for ticker in tickers:
    temp = pd.DataFrame(index=data.index)
    temp["ticker"] = ticker
    temp['Open'] = data[f"Open_{ticker}"]
    temp['High'] = data[f"High_{ticker}"]
    temp['Low'] = data[f"Low_{ticker}"]
    temp['Close'] = data[f"Close_{ticker}"]
    temp['Volume'] = data[f"Volume_{ticker}"]
    temp['ret'] = np.log(temp['Close']/temp['Close'].shift(1))
    temp['hl_range'] = (temp['High']-temp['Low'])/temp['Close']
    temp["oc_move"] = (temp["Close"] - temp["Open"]) / temp["Open"]
    temp["vol_change"] = temp["Volume"].pct_change()
    temp["volatility"] = (temp["ret"].rolling(10).std() * np.sqrt(252)).shift(-10)
    dfs.append(temp)

df = pd.concat(dfs)
df.dropna(inplace=True)
df = df.reset_index()
df.rename(columns={"Date": 'date'}, inplace=True)

df = df.sort_values(["ticker", "date"])

df.to_csv("OHLCV/unclean_OHLCV.csv")