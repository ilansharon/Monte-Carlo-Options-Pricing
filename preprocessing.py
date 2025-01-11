import yfinance as yf
import pandas as pd
from datetime import datetime

def fetchAtmData(ticker, expiration, Call):
    #ticker ex: "MSFT"
    #expiration ex: "2025-01-17"
    #Call Bool

    stock = yf.Ticker(ticker)
    underlyingPrice = stock.history(period="1d")['Close']
    underlyingPrice = underlyingPrice.iloc[-1]                              #get price of underlying asset
    
    allOptions = stock.option_chain(expiration)
    typedOptions = allOptions.calls if Call else allOptions.puts            #get option data of relevant type (Call or Put) into a dataframe 

    if Call:
        ATMRow = typedOptions[typedOptions['strike'] >= underlyingPrice]
        ATMRow = ATMRow.iloc[0]
    else:
        ATMRow = typedOptions[typedOptions['strike'] >= underlyingPrice]
        ATMRow = ATMRow.iloc[0]                                             #get data of relevant ATM row

    IV = ATMRow["impliedVolatility"]
    strike = ATMRow["strike"]                                               #select necessary values (Implied Volatility and Strike Price)

    return underlyingPrice, strike, IV

#NOTE TO SELF - add other functionality for different option types (ie, most interest, highest IV, etc.)


#function to get time to expiration
def timeToExpiration(expiration):
    #expiration in Year Month Day format
    current = datetime.now()
    expiration = datetime.strptime(expiration, "%Y-%m-%d")

    t = (expiration - current).days / 365.0             #get time difference and convert to years

    return t


#function go get risk free rate
def riskFreeRate():

    treasuryData = yf.Ticker("^IRX")                            # 13-week T-Bill rate index
    latestRate = treasuryData.history(period="1d")["Close"].iloc[-1]

    return latestRate / 100                                     # conver to decimal


#function to get all relevant option data at once
def optionData(ticker, expiration, Call):
    #for now, assume we use only atm
    u, s, v = fetchAtmData(ticker, expiration, Call)
    t = timeToExpiration(expiration)
    r = riskFreeRate()

    return u, s, v, t, r
