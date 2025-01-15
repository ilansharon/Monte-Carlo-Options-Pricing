import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def fetchAtmData(ticker, expiration, call):
    #ticker ex: "MSFT"
    #expiration ex: "2025-01-17"
    #Call Bool

    stock = yf.Ticker(ticker)
    underlyingPrice = stock.history(period="1d")['Close']
    underlyingPrice = underlyingPrice.iloc[-1]                              #get price of underlying asset
    
    allOptions = stock.option_chain(expiration)
    typedOptions = allOptions.calls if call else allOptions.puts            #get option data of relevant type (Call or Put) into a dataframe 

    if call:
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
def optionData(ticker, expiration, call, type):
    #for now, assume we use only atm
    if type == "ATM":
        s, k, v = fetchAtmData(ticker, expiration, call)
    t = timeToExpiration(expiration)
    r = riskFreeRate()

    return s, k, v, t, r

#get data for multiple strikes and expirations
def fetchMultiData(ticker, range):
    stock = yf.Ticker(ticker)
    expirations = np.array(stock.options)

    S0 = stock.history(period="1d")['Close']
    S0 = S0.iloc[-1]  

    strikes, prices, maturities, callFlags = [], [], [], []
    for date in expirations:
        T = timeToExpiration(date)
        if 1/52 < T < 1.5:  #over 1 week, less than 18 months
            roughData = stock.option_chain(date)
            roughCalls = roughData.calls
            roughPuts = roughData.puts

            # cut calls and puts to strikes within range
            calls = roughCalls[abs(S0 - roughCalls['strike']) < range]      
            puts = roughPuts[abs(S0 - roughPuts['strike']) < range]

            #seperate calls data into relevant arrays
            strikes.append(calls['strike'].to_numpy())
            prices.append(((calls['bid'] + calls['ask']) / 2).to_numpy())
            maturities.append(np.full(len(calls), T))
            callFlags.append(np.ones(len(calls), dtype = bool))

            #seperate puts data into relevant arrays
            strikes.append(puts['strike'].to_numpy())
            prices.append(((puts['bid'] + puts['ask']) / 2).to_numpy())
            maturities.append(np.full(len(puts), T))
            callFlags.append(np.zeros(len(puts), dtype = bool))

    strikes = np.concatenate(strikes)
    prices = np.concatenate(prices)
    maturities = np.concatenate(maturities)
    callFlags = np.concatenate(callFlags)


    return strikes, prices, maturities, callFlags




