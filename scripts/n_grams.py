import pandas as pd
import re
import os
from collections import Counter

# --- 1. 설정 ---
# [경로 수정] CSV 파일이 'data' 폴더 안에 있다고 가정
CSV_FILE = 'data/reddit_koreanfood.csv'


BASE_STOPWORDS_SET = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'but', 'if',
    'because', 'as', 'until', 'while','about', 'or', 'from', 'out',
    'any', 'some', 'my', 'korean', 'food', 'try', 'park', 'hello', 'hi',
    'recipe', 's', 't', 'm', 'what', 'on', 'in', 'at', 'by', 'for', 'with', 'about', 'of',
    'my', 'food', 'korean', 'is', 'it', 'the', 'then', 'beggining',
    'looking', 'to', 'korea', 'turned', 'bit', 'lot', 'top', 'advance',
    'how', 'where', 'some', 'that', 'this', 'made', 'make', 'do', 'so', 'kind', 'instead',
    's', 't', 'm', 've', 're', 'll', 'im', 'just', 'com', 'https', 'www', 'full', 'and',
    '이', '그', '저', '것', '수', '등', '들', '및', '제', 'd', 'k', 'r', 'haha', 'left', 'over'
    '의', '가', '이', '은', '는', '을', '를', '에', '와', '과', '도', 'not', 'ingredients', '전북',
    '너무', '정말', '진짜', '좀', '더', '오늘', '어떤', '무슨', 'like', 'can', 'world', 'cup', 'stadium'
    'homemade', 'help', 'question', 'recommendations', 'anyone', 'all', 'one', 'directly',
    'know', 'there', 'would', 'up', 'here', 'more', 'get', 'drink', 'everything', 
    'good', 'great', 'delicious', 'amazing', 'favorite', 'best', 'tasty', 'became', 'shoots',
    'love', 'really', 'very', 'rice', 'sauce', 'beef', 'pork', 'soup', 'too', 'both',
    'water', 'oil', 'sugar', 'salt', 'pepper', 'onion', 'garlic', 'time', 'even', 'okay',
    'first', 'video', 'post', 'today', 'night', 'dinner', 'lunch', 'breakfast', 'will',
    'go', 'eat', 'cook', 'cooking', 'tried', 'making', 'recommend', 'knows', 'bouncy', 'when',
    'jpg', 'png', 'reddit', 'preview', 'redd', 'https', 'www', 'com', 'org', 'want', 'home', 'after', 'find', 'used',
    'amp', 'x200b', 'nbsp', 've', 'm', 'll', 're', 's', 't', 'd', 'closed', 'baby'
}
new_stopwords_str = "everyday, side, dishes, stir, oct, 26, south, pop, demon, months, ago, 30, minutes, super, easy, nothing, fancy, weeks, still, use, mom, years, bon, appétit, demon, hunters, looks, like, seems, beats, room, temperature, also, called, temp, tastes, mix, hey, guys, something, else, bowl, asia, trader, joe, party"
NEW_STOPWORDS_SET = set(re.split(r'[ ,]+', new_stopwords_str))
STOPWORDS_SET = BASE_STOPWORDS_SET.union(NEW_STOPWORDS_SET)

print(f"--- 총 {len(STOPWORDS_SET)}개의 고유한 불용어를 사용합니다. ---")


# --- 2. N-Gram 분석 및 출력 함수 ---
def analyze_ngrams_for_comparison(csv_path):
    """
    CSV 파일을 읽어 1~5-gram을 "순수 발견" 방식으로 분석하고,
    Top 15 결과를 각각 print
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다.")
        print("스크립트가 루트 폴더에서 실행되고 있는지, 'data/reddit_koreanfood.csv' 경로가 맞는지 확인하세요.")
        return

    print(f"'{csv_path}' 로드 완료. 텍스트 정제를 시작합니다...")
    
    df['content'] = df['content'].fillna('')
    df['title'] = df['title'].fillna('')
    all_text = ' '.join(df['title'].str.lower() + ' ' + df['content'].str.lower())
    
    # --- 공통 필터링 함수 ---
    def filter_ngrams(ngrams_list):
        filtered_ngrams = []
        for phrase in ngrams_list:
            words_in_phrase = phrase.split()
            
            # 조건 1: 구문에 불용어, 1글자 단어, 숫자가 '하나라도' 포함되면 제외
            if any(w in STOPWORDS_SET or len(w) < 2 or w.isdigit() for w in words_in_phrase):
                continue
            
            filtered_ngrams.append(phrase)
        return filtered_ngrams

    # --- 1-gram (Unigram) 분석 ---
    unigrams = re.findall(r'\b(\w+)\b', all_text)
    filtered_unigrams = filter_ngrams(unigrams)
    unigram_counts = Counter(filtered_unigrams)
    
    print("\n--- [인사이트] 1-gram (단일 단어) Top 15 ---")
    df_unigram = pd.DataFrame(unigram_counts.most_common(15), columns=['Unigram', 'Frequency'])
    print(df_unigram.to_markdown(index=False, numalign="left", stralign="left"))

    # --- 2-gram (Bigram) 분석 ---
    bigrams = re.findall(r'\b(\w+ \w+)\b', all_text)
    filtered_bigrams = filter_ngrams(bigrams)
    bigram_counts = Counter(filtered_bigrams)
    
    print("\n--- [인사이트] 2-gram (2단어 구문) Top 15 ---")
    df_bigram = pd.DataFrame(bigram_counts.most_common(15), columns=['Bigram', 'Frequency'])
    print(df_bigram.to_markdown(index=False, numalign="left", stralign="left"))

    # --- 3-gram (Trigram) 분석 ---
    trigrams = re.findall(r'\b(\w+ \w+ \w+)\b', all_text)
    filtered_trigrams = filter_ngrams(trigrams)
    trigram_counts = Counter(filtered_trigrams)
    
    print("\n--- [인사이트] 3-gram (3단어 구문) Top 15 ---")
    df_trigram = pd.DataFrame(trigram_counts.most_common(15), columns=['Trigram', 'Frequency'])
    print(df_trigram.to_markdown(index=False, numalign="left", stralign="left"))

    # --- 4-gram (4-gram) 분석 ---
    four_grams = re.findall(r'\b(\w+ \w+ \w+ \w+)\b', all_text)
    filtered_4grams = filter_ngrams(four_grams)
    four_gram_counts = Counter(filtered_4grams)
    
    print("\n--- [인사이트] 4-gram (4단어 구문) Top 15 ---")
    df_4gram = pd.DataFrame(four_gram_counts.most_common(15), columns=['4-gram', 'Frequency'])
    print(df_4gram.to_markdown(index=False, numalign="left", stralign="left"))


# --- 3. 스크립트 실행 ---
if __name__ == "__main__":
    
    # 스크립트 파일 위치 기준으로 CSV 파일 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_file_path = os.path.join(project_root, CSV_FILE)

    analyze_ngrams_for_comparison(csv_file_path)