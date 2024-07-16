import requests
from bs4 import BeautifulSoup

def get_ticker_and_market_cap(url):
    # Отправляем HTTP запрос на получение страницы
    response = requests.get(url)
    # Проверим статус ответа
    if response.status_code != 200:
        print(f"Error: Unable to fetch the page, status code: {response.status_code}")
        return []

    # Парсим содержимое страницы с помощью BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Находим таблицу по селектору
    table = soup.find('table')
    
    # Проверяем, что таблица найдена
    if not table:
        print("Error: Table not found.")
        return []
    
    # Инициализируем список для хранения результатов
    results = []
    
    # Итерируемся по строкам таблицы
    for row in table.find('tbody').find_all('tr'):
        # Извлекаем тикер
        ticker_element = row.select_one('td:nth-child(3) > a > div > div > div > p')
        if ticker_element:
            ticker = ticker_element.text.strip()
        else:
            ticker = "N/A"
    
    return results

# URL страницы CoinMarketCap
url = 'https://coinmarketcap.com/trending-cryptocurrencies/'
tickers_and_market_caps = get_ticker_and_market_cap(url)

# Печатаем результаты
for ticker, market_cap in tickers_and_market_caps:
    print(f"Тикер: {ticker}, Рыночная капитализация: {market_cap}")