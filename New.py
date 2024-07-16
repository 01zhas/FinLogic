import requests
from bs4 import BeautifulSoup
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_coinmarketcap_tickers():
    url = 'https://coinmarketcap.com/trending-cryptocurrencies/'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: Unable to fetch the page, status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    tickers = []
    
    table = soup.find('table')
    if not table:
        print("Error: Table not found.")
        return []

    for row in table.find('tbody').find_all('tr'):
        ticker_element = row.select_one('td:nth-child(3) > a > div > div > div > p')
        if ticker_element:
            tickers.append(ticker_element.text.strip() + "-USD")

    return tickers

def get_yahoo_tickers():
    url = 'https://finance.yahoo.com/trending-tickers/'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: Unable to fetch the page, status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    tickers = []

    table = soup.find('tbody')
    if not table:
        print("Error: Table with ID 'list-res-table' not found.")
        return []

    for row in table.find_all('tr'):
        ticker = row.find('td', {'aria-label': 'Symbol'}).text.strip()
        tickers.append(ticker)

    return tickers

def get_all_tickers():
    coinmarketcap_tickers = get_coinmarketcap_tickers()
    yahoo_tickers = get_yahoo_tickers()
    
    all_tickers = list(set(coinmarketcap_tickers + yahoo_tickers))  # Удаление дубликатов
    return all_tickers

def filter_tickers(tickers):
    filtered_tickers = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap')
            volume = stock.info.get('volume')
            if market_cap and market_cap > 1e9 and volume and volume > 1e6:
                filtered_tickers.append(ticker)
        except Exception:
            continue  # Игнорируем ошибки и продолжаем обработку следующих тикеров
    return filtered_tickers

def calculate_sortino_ratio(ticker, start_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date)
    
    
    if hist.empty or hist.index[0].tz_convert(None) > start_date + timedelta(days=5):
        return None

    returns = hist['Close'].pct_change().dropna()
    downside_returns = returns[returns < 0]
    expected_return = returns.mean()
    downside_std = downside_returns.std()

    if downside_std == 0:
        return None 

    sortino_ratio = expected_return / downside_std
    return sortino_ratio

def get_top_4_by_sortino(filtered_tickers):
    start_date = (datetime.now() - timedelta(days=3*365)).replace(tzinfo=None)  
    sortino_ratios = {}
    for ticker in filtered_tickers:
        ratio = calculate_sortino_ratio(ticker, start_date)
        if ratio is not None:
            sortino_ratios[ticker] = ratio
    
    sorted_tickers = sorted(sortino_ratios.items(), key=lambda x: x[1], reverse=True)
    
    crypto = None
    stocks = []
    for ticker, ratio in sorted_tickers:
        if crypto is None and "-USD" in ticker:
            crypto = ticker
        elif len(stocks) < 3:
            stocks.append(ticker)
        if crypto and len(stocks) == 3:
            break

    top_4_tickers = []
    if crypto:
        top_4_tickers.append(crypto)
    top_4_tickers.extend(stocks)

    return top_4_tickers

all_tickers = get_all_tickers()
filtered_tickers = filter_tickers(all_tickers)
top_4_tickers = get_top_4_by_sortino(filtered_tickers)
print("Топ 4 тикеров по коэффициенту Сортино:", top_4_tickers)