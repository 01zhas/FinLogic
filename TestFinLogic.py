import unittest
from unittest.mock import patch
import pandas as pd
from datetime import datetime, timedelta
from FinLogic import FinLogic  # Импортируем ваш класс FinLogic

class TestFinLogic(unittest.TestCase):
    
    @patch.object(FinLogic, 'fetch_data')
    @patch.object(FinLogic, 'get_conversion_rate')
    @patch.object(FinLogic, 'get_coinmarketcap_tickers')
    @patch.object(FinLogic, 'get_yahoo_tickers')
    def test_optimize_portfolio(self, mock_yahoo_tickers, mock_coinmarketcap_tickers, mock_conversion_rate, mock_fetch_data):
        # Определяем фиктивные значения, которые будут возвращены заглушками
        mock_yahoo_tickers.return_value = ['AAPL', 'MSFT', 'GOOGL']
        mock_coinmarketcap_tickers.return_value = ['BTC-USD', 'ETH-USD']
        mock_conversion_rate.return_value = 425.0  # Пример: 1 USD = 425 KZT
        
        # Создаем фиктивные данные для fetch_data
        date_range = pd.date_range(start='2020-01-01', end='2023-01-01')
        mock_data = pd.DataFrame(index=date_range, data={
            'AAPL': 150 + 0.01 * range(len(date_range)),
            'MSFT': 200 + 0.01 * range(len(date_range)),
            'GOOGL': 2500 + 0.01 * range(len(date_range)),
            'BTC-USD': 50000 + 10 * range(len(date_range)),
            'ETH-USD': 3000 + 1 * range(len(date_range)),
        })
        mock_fetch_data.return_value = mock_data

        # Создаем экземпляр FinLogic и тестируем метод optimize_portfolio
        fin_logic = FinLogic(data_dir='data', portfolio_file='portfolio.csv')
        new_investment = 140000
        fin_logic.optimize_portfolio(new_investment)

        # Здесь вы можете добавить дополнительные проверки для проверки корректности оптимизации
        # Например, проверка финальных аллокаций, сохранения файла и т.д.

if __name__ == '__main__':
    unittest.main()