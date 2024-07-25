import requests
from bs4 import BeautifulSoup
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
import warnings
import json
import os
import ta
import cvxpy as cp

warnings.filterwarnings("ignore")

class FinLogic:
    def __init__(self, data_dir='data', portfolio_file='portfolio.csv', cache_file='cache.json'):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.data_dir = data_dir
        self.portfolio_file = portfolio_file
        self.current_portfolio = self.load_portfolio()
        self.default_start_date = (datetime.today() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.data_store = {}
        # self.load_data(list(self.current_portfolio.keys()))

    def load_portfolio(self):
        print("Загрузка портфеля...")
        portfolio = pd.read_csv(self.portfolio_file, index_col='Ticker').to_dict()['Value']
        print("Портфель загружен.")
        return portfolio

    def fetch_data(self, tickers):
        print(f"Загрузка данных для тикеров: {tickers}...")
        data = yf.download(tickers, start=self.default_start_date, end=self.end_date)['Close']
        print(f"Данные для тикеров {tickers} загружены.")
        return data

    def load_data(self, tickers):
        print(f"Загрузка данных для тикеров: {tickers}...")
        full_index = pd.date_range(start=self.default_start_date, end=self.end_date)
        for ticker in tickers:
            data_path = os.path.join(self.data_dir, f'{ticker}.csv')
            if os.path.exists(data_path):
                print(f"Загрузка данных из файла для {ticker}...")
                data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
            else:
                data = self.fetch_data([ticker])
                data.to_csv(data_path)
            data = data.loc[~data.index.duplicated(keep='first')]
            data = data.reindex(full_index).fillna(method='ffill').fillna(method='bfill')
            self.data_store[ticker] = data
            print(f"Данные для {ticker} загружены.")
        print("Все данные загружены.")

    def get_conversion_rate(self):
        print("Получение курса конверсии USD to KZT...")
        url = 'https://api.exchangerate-api.com/v4/latest/USD'
        response = requests.get(url)
        data = response.json()
        rate = data['rates']['KZT']
        print(f"Курс конверсии: 1 USD = {rate} KZT")
        return rate

    def optimize_portfolio(self, new_investment):
        print("Оптимизация портфеля...")
        self.conversion_rate = self.get_conversion_rate()

        all_tickers = self.get_all_tickers()
        filtered_tickers = self.filter_tickers(all_tickers, new_investment)
        
        top_4_tickers = self.get_top_4_by_sortino(filtered_tickers)
        
        if not top_4_tickers:
            print("Нет доступных тикеров для оптимизации.")
            return
        
        stock_tickers = [ticker for ticker in top_4_tickers if not ticker.endswith("-USD")]
        crypto_tickers = [ticker for ticker in top_4_tickers if ticker.endswith("-USD")]
        
        self.load_data(top_4_tickers)
        
        data_stocks = pd.concat([self.data_store[ticker] for ticker in stock_tickers], axis=1)
        data_crypto = pd.concat([self.data_store[ticker] for ticker in crypto_tickers], axis=1)

        returns_stocks = data_stocks.pct_change().dropna()
        returns_crypto = data_crypto.pct_change().dropna()

        mean_returns_stocks = returns_stocks.mean().values
        mean_returns_crypto = returns_crypto.mean().values

        cov_matrix_stocks = returns_stocks.cov().values
        cov_matrix_crypto = returns_crypto.cov().values

        num_assets_stocks = len(stock_tickers)
        num_assets_crypto = len(crypto_tickers)

        shares_stocks = cp.Variable(num_assets_stocks, integer=True)
        shares_crypto = cp.Variable(num_assets_crypto)

        binary_stocks = cp.Variable(num_assets_stocks, boolean=True)
        binary_crypto = cp.Variable(num_assets_crypto, boolean=True)

        current_prices_stocks = data_stocks.iloc[-1].values
        current_prices_crypto = data_crypto.iloc[-1].values

        objective = cp.Maximize(mean_returns_stocks @ shares_stocks + mean_returns_crypto @ shares_crypto)

        risk_tolerance = 0.02  # Примерное допустимое значение стандартного отклонения (2%)

        constraints = [
            shares_stocks >= 0,
            shares_crypto >= 0,
            shares_stocks <= binary_stocks * 1e6,  # Устанавливаем верхнюю границу для акций
            shares_crypto <= binary_crypto * 1e6,  # Устанавливаем верхнюю границу для криптовалют
            shares_stocks @ current_prices_stocks + shares_crypto @ current_prices_crypto == new_investment / self.conversion_rate,
            shares_stocks @ current_prices_stocks >= binary_stocks * 5000 / self.conversion_rate,
            shares_crypto @ current_prices_crypto >= binary_crypto * 5000 / self.conversion_rate,
            cp.quad_form(shares_stocks, cov_matrix_stocks) + cp.quad_form(shares_crypto, cov_matrix_crypto) <= risk_tolerance
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI)

        investment_amounts_stocks = shares_stocks.value * current_prices_stocks
        investment_amounts_crypto = shares_crypto.value * current_prices_crypto

        stocks_allocations = {}
        crypto_allocations = {}
        stock_quantities = {}
        crypto_quantities = {}
        
        for ticker, num_shares, investment in zip(stock_tickers, shares_stocks.value, investment_amounts_stocks):
            if num_shares > 0:
                stocks_allocations[ticker] = investment
                stock_quantities[ticker] = num_shares
                
        for ticker, num_shares, investment in zip(crypto_tickers, shares_crypto.value, investment_amounts_crypto):
            if num_shares > 0:
                crypto_allocations[ticker] = investment
                crypto_quantities[ticker] = num_shares

        def print_allocations(allocations, quantities, category):
            total_value = sum(allocations.values())
            print(f"\n{category} инвестиции:")
            for ticker, investment in allocations.items():
                investment_KZT = investment * self.conversion_rate
                change = round(investment - self.current_portfolio.get(ticker, 0), 2)
                final_value_rounded = round(investment, 2)
                final_value_usd = round(final_value_rounded, 2)
                change_usd = round(change, 2)
                quantity = quantities[ticker]
                print(f"{ticker}: {quantity} единиц, {final_value_rounded}$ ({investment_KZT:.2f}₸) \033[92m{'+' if change >= 0 else ''}{change_usd}$ ({change * self.conversion_rate:.2f}₸)\033[0m")
            print(f"Общая сумма в {category.lower()}: {round(total_value, 2)}$ ({round(total_value * self.conversion_rate, 2)}₸)")

        print_allocations(stocks_allocations, stock_quantities, "Акции")
        print_allocations(crypto_allocations, crypto_quantities, "Криптовалюты")

        total_investment_value = sum(investment_amounts_stocks) + sum(investment_amounts_crypto)
        print(f"\nОбщий объем инвестиций: {round(total_investment_value, 2)}$ ({round(total_investment_value * self.conversion_rate, 2)}₸)")

        if input("Перезаписать portfolio.csv новыми значениями? (y/n): ").lower() == 'y':
            new_portfolio = pd.DataFrame(list(stocks_allocations.items()) + list(crypto_allocations.items()), columns=['Ticker', 'Value'])
            new_portfolio['Value'] = new_portfolio['Value'].round(2)  # Округление значений
            new_portfolio.to_csv(self.portfolio_file, index=False)
            print("Файл portfolio.csv обновлен.")
        else:
            print("Обновление файла отменено.")


    def get_coinmarketcap_tickers(self):
        print("Получение трендовых криптовалют с CoinMarketCap...")
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

        print(f"Тикеры с CoinMarketCap: {tickers}")
        return tickers

    def get_yahoo_tickers(self):
        print("Получение трендовых акций с Yahoo Finance...")
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

        print(f"Тикеры с Yahoo Finance: {tickers}")
        return tickers

    def get_yahoo_tickers1(self):
        print("Получение трендовых акций с Yahoo Finance...")
        url = 'https://finance.yahoo.com/crypto?offset=0&count=100'
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

        print(f"Тикеры с Yahoo Finance: {tickers}")
        return tickers

    def get_yahoo_tickers2(self):
        print("Получение трендовых акций с Yahoo Finance...")
        url = 'https://finance.yahoo.com/most-active?offset=0&count=100'
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

        print(f"Тикеры с Yahoo Finance: {tickers}")
        return tickers

    def get_all_tickers(self):
        print("Получение всех тикеров...")
        coinmarketcap_tickers = self.get_coinmarketcap_tickers()
        yahoo_tickers = self.get_yahoo_tickers()
        yahoo_tickers1 = self.get_yahoo_tickers1()
        yahoo_tickers2 = self.get_yahoo_tickers2()


        
        all_tickers = list(set(coinmarketcap_tickers + yahoo_tickers + yahoo_tickers1 + yahoo_tickers2))  # Удаление дубликатов
        print(f"Все тикеры: {all_tickers}")
        return all_tickers

    def filter_tickers(self, tickers, new_investment):
        print("Фильтрация тикеров...")
        filtered_tickers = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                market_cap = stock.info.get('marketCap')
                volume = stock.info.get('volume')
                current_price = stock.history(period="1d")['Close'].iloc[0]

                # Check if the market cap and volume meet the criteria
                if market_cap and market_cap > 1e9 and volume and volume > 1e6:
                    # For stocks, check if the price does not exceed 40% of the new investment
                    if not ticker.endswith("-USD") and current_price <= 0.4 * (new_investment / self.conversion_rate):
                        filtered_tickers.append(ticker)
                    # For cryptocurrencies, just add them without the price check
                    elif ticker.endswith("-USD"):
                        filtered_tickers.append(ticker)
            except Exception as e:
                print(f"Ошибка при обработке тикера {ticker}: {e}")
                continue  # Игнорируем ошибки и продолжаем обработку следующих тикеров
        print(f"Отфильтрованные тикеры: {filtered_tickers}")
        return filtered_tickers

    def calculate_sortino_ratio(self, ticker, start_date):
        print(f"Расчет коэффициента Сортино для {ticker}...")
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
        print(f"Коэффициент Сортино для {ticker}: {sortino_ratio}")
        return sortino_ratio

    def calculate_technical_indicators(self, ticker, start_date):
        print(f"Расчет технических индикаторов для {ticker}...")
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date)
        
        if hist.empty or hist.index[0].tz_convert(None) > start_date + timedelta(days=5):
            return None
        hist['SMA_50'] = ta.trend.sma_indicator(hist['Close'], window=50)
        hist['SMA_200'] = ta.trend.sma_indicator(hist['Close'], window=200)
        hist['RSI'] = ta.momentum.rsi(hist['Close'], window=14)

        print(f"Технические индикаторы для {ticker}: SMA_50 = {hist['SMA_50'].iloc[-1]}, SMA_200 = {hist['SMA_200'].iloc[-1]}, RSI = {hist['RSI'].iloc[-1]}")
        return hist

    def get_top_4_by_sortino(self, filtered_tickers):
        print("Получение топ-4 тикеров по коэффициенту Сортино с учетом технических индикаторов...")
        start_date = (datetime.now() - timedelta(days=3*365)).replace(tzinfo=None)  
        sortino_ratios = {}
        tech_indicators = {}
        
        for ticker in filtered_tickers:
            ratio = self.calculate_sortino_ratio(ticker, start_date)
            tech_data = self.calculate_technical_indicators(ticker, start_date)
            
            if ratio is not None and tech_data is not None:
                if tech_data['SMA_50'].iloc[-1] > tech_data['SMA_200'].iloc[-1] and tech_data['RSI'].iloc[-1] < 70:
                    sortino_ratios[ticker] = ratio
                    tech_indicators[ticker] = tech_data.iloc[-1]  # Last row of data
        
        sorted_tickers = sorted(sortino_ratios.items(), key=lambda x: x[1], reverse=True)
        
        top_4_tickers = []
        crypto = None
        stocks = []


        for ticker, ratio in sorted_tickers:
            if self.check_asset_availability(ticker):
                top_4_tickers.append(ticker)

        # for ticker, ratio in sorted_tickers:
        #     if crypto is None and "-USD" in ticker:
        #         if self.check_asset_availability(ticker):
        #             crypto = ticker
        #     elif len(stocks) < 3:
        #         if self.check_asset_availability(ticker):
        #             stocks.append(ticker)
        #     if crypto and len(stocks) == 3:
        #         break

        # if crypto:
        #     top_4_tickers.append(crypto)
        # top_4_tickers.extend(stocks)

        # for ticker in top_4_tickers:
        #     tech_data = tech_indicators[ticker]
        #     print(f"{ticker}: Sortino Ratio: {sortino_ratios[ticker]}, SMA_50: {tech_data['SMA_50']}, SMA_200: {tech_data['SMA_200']}, RSI: {tech_data['RSI']}")

        return top_4_tickers

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as file:
                return json.load(file)
        return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as file:
            json.dump(self.cache, file)

    def check_asset_availability(self, ticker):
        if ticker in self.cache:
            available = self.cache[ticker]
            print(f"Актив {ticker} {'доступен' if available else 'недоступен'} для приобретения (из кэша).")
            return available

        response = input(f"Можете ли вы приобрести {ticker}? (y/n): ").strip().lower()
        available = response == 'y'
        self.cache[ticker] = available
        self.save_cache()
        print(f"Актив {ticker} {'доступен' if available else 'недоступен'} для приобретения.")
        return available

fin_logic = FinLogic('data', 'portfolio.csv')
new_investment = 355800
final_allocations = fin_logic.optimize_portfolio(new_investment)