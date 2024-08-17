import requests
from bs4 import BeautifulSoup
import json
import re

def extract_text_from_url(url):
    # 웹페이지의 전체 텍스트를 가져오기
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator=' ')
    else:
        return None

def clean_text(text):
    # 텍스트 정제
    text = re.sub(r'\s+', ' ', text)  # 다중 공백을 단일 공백으로
    text = re.sub(r'[^A-Za-z가-힣0-9 ]+', '', text)  # 문자, 숫자, 한글만 남기기
    return text.strip()

# 기존 JSON 파일 로드
with open('news_data.json', 'r', encoding='utf-8') as json_file:
    news_items = json.load(json_file)

# 정제된 텍스트만 저장할 리스트
detailed_news_texts = []

for item in news_items:
    article_url = item.get('link', item.get('originallink'))
    
    # 전체 텍스트 추출
    full_text = extract_text_from_url(article_url)
    
    if full_text:
        # 텍스트 정제
        cleaned_text = clean_text(full_text)
        detailed_news_texts.append({"full_text": cleaned_text})
    else:
        detailed_news_texts.append({"full_text": "텍스트를 가져올 수 없습니다."})

# 추출된 전체 텍스트만을 새로운 JSON 파일에 저장
with open('full_texts_only.json', 'w', encoding='utf-8') as json_file:
    json.dump(detailed_news_texts, json_file, ensure_ascii=False, indent=4)

print("정제된 전체 텍스트가 'full_texts_only.json' 파일에 저장되었습니다.")
