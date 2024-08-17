import requests
import json

# 네이버 API 설정
NAVER_CLIENT_ID = 'API를 네이버를 통해 발급 받아야 합니다'
NAVER_CLIENT_SECRET = 'API를 네이버를 통해 발급 받아야 합니다'

# 뉴스 데이터 검색 함수 정의
def get_latest_news(query, display=100):
    url = f'https://openapi.naver.com/v1/search/news.json?query={query}&display={display}&sort=date'
    headers = {
        'X-Naver-Client-Id': NAVER_CLIENT_ID,
        'X-Naver-Client-Secret': NAVER_CLIENT_SECRET
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data['items']
    else:
        print(f"Error: API request failed with status code {response.status_code}")
        return []

# JSON 파일로 저장하는 함수 정의
def save_news_to_json(news_data, filename="news_data.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(news_data, f, ensure_ascii=False, indent=4)
    print(f"뉴스 데이터를 '{filename}' 파일에 저장했습니다.")

# 실행 예시
query = "비트코인"
news_items = get_latest_news(query)
save_news_to_json(news_items, "news_data.json")
