AI DATA 공모전을 하는 과정에서 네이버 API를 활용하여 NLP 모델을 사용할 수 있을까에서 시작했습니다. (결과적으로 실패함...)

네이버 API를 통해 뉴스 데이터를 수집!!! > news.py ( https://developers.naver.com/docs/serviceapi/search/news/news.md ) API 발급 받을 수 있는 링크 첨부합니다.

뉴스 데이터의 json 파일을 살펴보면 description 부분을 통해 전체가 아닌 요약된 뉴스가 나온다는 것을 알 수 있어서 전체 텍스트를 추출하는 코드를 추가 > news2.py

네이버 API만으로는 NLP 모델을 사용할 수 없어서 API의 1. 감정분석과 2. 요약 후 키워드 기반 처리 3. bert 활용 > API3.py ( https://www.ncloud.com/ ) API 발급 받을 수 있는 링크 첨부합니다.

뉴스 데이터를 통해 비트코인 가격의 상승, 하락, 횡보 3가지 중 하나를 예측하는? 키워드 분류? 하는 모델을 만들고 싶었다 위의 3가지 분류 이후 가중치를 통해 최종 결과 값을 도출함
