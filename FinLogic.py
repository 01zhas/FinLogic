import requests
from bs4 import BeautifulSoup
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
import warnings
import os
import ta

warnings.filterwarnings("ignore")

class FinLogic:
    def __init__(self, data_dir='data', portfolio_file='portfolio.csv'):
        self.data_dir = data_dir
        self.portfolio_file = portfolio_file
        self.current_portfolio = self.load_portfolio()
        self.default_start_date = (datetime.today() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.data_store = {}
        self.load_data(list(self.current_portfolio.keys()))

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
        conversion_rate = self.get_conversion_rate()

        all_tickers = self.get_all_tickers()
        filtered_tickers = self.filter_tickers(all_tickers)
        
        top_4_tickers = self.get_top_4_by_sortino(filtered_tickers)
        
        for ticker in top_4_tickers:
            if ticker not in self.current_portfolio:
                self.current_portfolio[ticker] = 0  
        
        self.load_data(list(self.current_portfolio.keys()))
        
        historical_data = pd.concat(self.data_store.values(), axis=1)
        historical_data.columns = [ticker for ticker in self.data_store.keys()]
        mu = expected_returns.mean_historical_return(historical_data)
        S = risk_models.sample_cov(historical_data)
        
        ef = EfficientFrontier(mu, S, weight_bounds=(0.01, 1))
        current_portfolio_value = sum(self.current_portfolio.values())
        total_value = current_portfolio_value + new_investment
        current_weights = {asset: value / total_value for asset, value in self.current_portfolio.items()}

        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        final_allocations = {}
        for asset, weight in cleaned_weights.items():
            if self.current_portfolio.get(asset, 0) == 0:
                final_allocations[asset] = weight * new_investment
            else:
                final_allocations[asset] = self.current_portfolio[asset] + (weight * new_investment)

        stocks_allocations = {}
        crypto_allocations = {}
        for asset, final_value in final_allocations.items():
            if "-USD" in asset:
                crypto_allocations[asset] = final_value
            else:
                stocks_allocations[asset] = final_value

        def print_allocations(allocations, category):
            total_value = sum(allocations.values())
            print(f"\n{category} инвестиции:")
            for asset, final_value in allocations.items():
                change = round(final_value - self.current_portfolio.get(asset, 0), 2)
                final_value_rounded = round(final_value, 2)
                final_value_usd = round(final_value_rounded / conversion_rate, 2)
                change_usd = round(change / conversion_rate, 2)
                print(f"{asset}: {final_value_rounded}₸ ({final_value_usd}$) \033[92m{'+' if change >= 0 else ''}{change}₸ ({change_usd}$)\033[0m")
            print(f"Общая сумма в {category.lower()}: {round(total_value, 2)}₸ ({round(total_value / conversion_rate, 2)}$)")

        print_allocations(stocks_allocations, "Акции")
        print_allocations(crypto_allocations, "Криптовалюты")

        total_investment_value = sum(final_allocations.values())
        print(f"\nОбщий объем инвестиций: {round(total_investment_value, 2)}₸ ({round(total_investment_value / conversion_rate, 2)}$)")

        if input("Перезаписать portfolio.csv новыми значениями? (y/n): ").lower() == 'y':
            new_portfolio = pd.DataFrame(list(final_allocations.items()), columns=['Ticker', 'Value'])
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

    def get_all_tickers(self):
        print("Получение всех тикеров...")
        coinmarketcap_tickers = self.get_coinmarketcap_tickers()
        yahoo_tickers = self.get_yahoo_tickers()
        
        all_tickers = list(set(coinmarketcap_tickers + yahoo_tickers))  # Удаление дубликатов
        print(f"Все тикеры: {all_tickers}")
        return all_tickers

    def filter_tickers(self, tickers):
        print("Фильтрация тикеров...")
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
            if crypto is None and "-USD" in ticker:
                if self.check_asset_availability(ticker):
                    crypto = ticker
            elif len(stocks) < 3:
                if self.check_asset_availability(ticker):
                    stocks.append(ticker)
            if crypto and len(stocks) == 3:
                break

        if crypto:
            top_4_tickers.append(crypto)
        top_4_tickers.extend(stocks)

        for ticker in top_4_tickers:
            tech_data = tech_indicators[ticker]
            print(f"{ticker}: Sortino Ratio: {sortino_ratios[ticker]}, SMA_50: {tech_data['SMA_50']}, SMA_200: {tech_data['SMA_200']}, RSI: {tech_data['RSI']}")

        return top_4_tickers

    def check_asset_availability(self, ticker):
        response = input(f"Можете ли вы приобрести {ticker}? (y/n): ").strip().lower()
        available = response == 'y'
        print(f"Актив {ticker} {'доступен' if available else 'недоступен'} для приобретения.")
        return available
    

fin_logic = FinLogic('data', 'portfolio.csv')
new_investment = 140000
final_allocations = fin_logic.optimize_portfolio(new_investment)