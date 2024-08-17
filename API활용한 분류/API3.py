import json
import requests
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim

# 네이버 API 설정
NAVER_API_KEY_ID = '네이버 플랫폼 API 필요'
NAVER_API_KEY = '네이버 플랫폼 API 필요'

# 1. 네이버 API 감정 분석
def analyze_sentiment(text):
    url = "https://naveropenapi.apigw.ntruss.com/sentiment-analysis/v1/analyze"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_API_KEY_ID,
        "X-NCP-APIGW-API-KEY": NAVER_API_KEY,
        "Content-Type": "application/json"
    }
    data = {"content": text}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        sentiment = response.json()['document']['sentiment']
        return sentiment  # 'positive', 'negative', 'neutral'
    else:
        return "neutral"

# 2. 네이버 API 요약
def summarize_text(text):
    url = "https://naveropenapi.apigw.ntruss.com/text-summary/v1/summarize"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_API_KEY_ID,
        "X-NCP-APIGW-API-KEY": NAVER_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "document": {"content": text},
        "option": {"language": "ko", "model": "general", "tone": 2, "summaryCount": 3}
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        summary = response.json().get('summary', "")
        return summary
    else:
        return ""

# 3. 키워드 기반 분석
def extract_keywords(texts, top_n=10):
    if not texts or not any(texts):
        return []

    # 문서가 하나인 경우 max_df와 min_df 설정을 조정
    if len(texts) == 1:
        vectorizer = TfidfVectorizer(max_features=1000)
    else:
        vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, max_features=1000)
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    sum_tfidf = tfidf_matrix.sum(axis=0)
    keywords = [(feature_names[col], sum_tfidf[0, col]) for col in sum_tfidf.nonzero()[1]]
    keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]
    return [keyword for keyword, score in keywords]

def classify_keywords(keywords):
    if not keywords:
        return 'neutral', "No significant keywords identified"

    positive_words = ['상승', '호재', '긍정적']
    negative_words = ['하락', '악재', '부정적']

    score = 0
    explanation = []
    for keyword in keywords:
        if keyword in positive_words:
            score += 1
            explanation.append(f"Keyword '{keyword}' indicates a positive trend.")
        elif keyword in negative_words:
            score -= 1
            explanation.append(f"Keyword '{keyword}' indicates a negative trend.")

    if score > 0:
        return 'positive', " ".join(explanation)
    elif score < 0:
        return 'negative', " ".join(explanation)
    else:
        return 'neutral', "Keywords indicate a neutral trend."

# 4. BERT 기반 분석 및 모델 설정
class BitcoinPredictor(nn.Module):
    def __init__(self):
        super(BitcoinPredictor, self).__init__()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 3)  # 상승, 동결, 하락 (positive, neutral, negative)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
model_bert = BertModel.from_pretrained('monologg/kobert')
predictor_model = BitcoinPredictor()

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    outputs = model_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# 모델 학습 함수
def train_model(predictor_model, train_data, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(predictor_model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, label in train_data:
            optimizer.zero_grad()
            outputs = predictor_model(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}')

# 뉴스 데이터 필터링
def filter_news_by_length(news_items, min_length=50):
    filtered_news = []
    for item in news_items:
        if len(item.split()) >= min_length:
            filtered_news.append(item)
    return filtered_news

# 뉴스 데이터 처리
def process_news_data(news_items):
    results = []

    for news in news_items:
        news_result = {"text": news}

        # 1. 네이버 API 감정 분석
        sentiment = analyze_sentiment(news)
        news_result["sentiment_api"] = sentiment

        # 2. 네이버 API 요약 및 키워드 분석
        summary = summarize_text(news)
        if not summary:
            summary = news  # 요약 실패 시 원문 사용
        news_result["summary"] = summary

        keywords = extract_keywords([summary])
        keyword_sentiment, keyword_explanation = classify_keywords(keywords)
        news_result["keyword_classification"] = keyword_sentiment
        news_result["keyword_explanation"] = keyword_explanation

        # 3. BERT 기반 분석 (전체 뉴스 텍스트를 기반으로)
        embedding = get_bert_embeddings(news).detach()
        prediction = predictor_model(embedding)
        predicted_label = torch.argmax(prediction).item()
        labels = ["positive", "neutral", "negative"]
        news_result["bert_prediction"] = labels[predicted_label]

        results.append(news_result)

    return results

# 결과를 파일로 저장
def save_results_to_file(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

# 최종 방향성 판단
def determine_final_direction(results):
    final_score = 0
    for result in results:
        if result["sentiment_api"] == "positive" or result["bert_prediction"] == "positive" or result["keyword_classification"] == "positive":
            final_score += 1
        elif result["sentiment_api"] == "negative" or result["bert_prediction"] == "negative" or result["keyword_classification"] == "negative":
            final_score -= 1

    if final_score > 0:
        return "상승"
    elif final_score < 0:
        return "하락"
    else:
        return "중립"

# JSON 파일에서 뉴스 데이터 로드
def load_news_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        news_data = json.load(file)
    return [item['full_text'] for item in news_data if 'full_text' in item]

# 실행
news_items = load_news_data('full_texts_only.json')
filtered_news_items = filter_news_by_length(news_items, min_length=50)

# 뉴스 데이터 처리
results = process_news_data(filtered_news_items)

# 예시 학습 데이터 (처리된 뉴스 데이터를 기반으로 훈련 데이터 생성)
train_data = [
    (get_bert_embeddings(news['text']).detach(), torch.tensor([0 if news['sentiment_api'] == 'positive' else 1 if news['sentiment_api'] == 'neutral' else 2]))
    for news in results
]

# 모델 학습
train_model(predictor_model, train_data)

# 결과 파일로 저장
save_results_to_file(results, 'news_analysis_results.csv')

# 최종 방향성 판단
final_direction = determine_final_direction(results)

print(f"최종 비트코인 가격 방향성: {final_direction}")
