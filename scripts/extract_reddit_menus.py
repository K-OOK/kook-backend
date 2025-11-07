import pandas as pd
import re
import sqlite3
import os
from collections import Counter

# --- 1. 설정 ---
DB_FILE = 'kfood_recipes.db'
TABLE_NAME = 'hot_recipes'
CSV_FILE = 'data/reddit_koreanfood.csv'


STOPWORDS_SET = {
    'it', 'something', 'ourselves', 'on', 'still', 'were', 'salt', 'closed', 'making', 'out', 'also', 'recommendations', 'cup', 'eat', 'try', '저', 'im', 'more', 'was', 'hi', 'by', 'my', 'food', 'to', 'lot', 'www', 'became', 'knows', 'mix', '수', 'can', 'here', 'drink', 'who', 'would', 'has', 'them', 'does', 'find', 'amp', '은', 'all', 'beggining', 'weeks', 'ours', 'best', 'tried', 'shoots', 'hunters', 'after', 'our', 'or', '를', 'go', 'bouncy', 'appétit', 'doing', 'delicious', 'what', 'park', 'post', 'fancy', 'because', 'dishes', 'about', 'k', 'd', 'rice', 're', 'guys', '너무', 'from', 'll', 'like', 'amazing', 'some', 'recipe', 'at', 'both', 't', 'okay', 'reddit', 'not', '더', 'else', 'use', 'pop', 'just', 'asia', 'are', 'kind', 'whom', 'favorite', 'korean', 'even', 'me', 'everyday', 'where', 'i', 'recommend', 'joe', 'onion', 'hey', 'with', 'pork', 'have', 'their', '그', 'world', 'that', 'dinner', 'baby', 'turned', 'bowl', '전북', 'really', 'myself', 'png', 'nothing', 'is', 'being', 'home', '과', 'anyone', 'ingredients', 'ago', '는', 'tasty', 'love', 'be', 'called', '들', 'bon', 'mom', 'her', 'this', 'but', 'his', 'you', 'today', 'minutes', 'tastes', 'looking', 'garlic', '및', 'beef', 'cooking', 'super', '도', 'stadiumhomemade', '와', 'sauce', 'and', 'good', 'water', 'video', 'these', 'first', '정말', 'when', 'everything', 'beats', 'we', 'months', 'get', 'demon', 'while', 'top', '시금치', 'macaroni', 'm', 'great', 'been', 'redd', 'bit', 'of', 'as', '오늘', '좀', 'too', '에', 'up', 'make', '것', 'made', 'preview', 'org', 'full', 'the', 'hello', 'breakfast', 'x200b', 'soup', '무슨', 'sugar', 'will', 'years', '을', 'httpss', 'easy', '30', 'know', 'oil', 'temp', 'left', 'nbsp', 'directly', 'looks', 'she', 'he', 'which', 'do', 'help', 'used', '제', '진짜', 'advance', 'seems', 'very', 'having', '등', 'instead', 'trader', 'korea', 'lunch', 'cook', 'until', 'in', 've', 'any', 'one', 'am', 'night', '26', 'stir', 'those', 'then', 'for', 'how', 'side', 'south', 'room', 'him', 'they', 'its', 'pepper', 'haha', 'over의', 'com', 'r', 'so', 'party', 'temperature', 'your', 'did', '이', 's', 'time', 'want', 'had', 'question', 'there', 'jpg', 'oct', '어떤', 'if'
}
print(f"--- 총 {len(STOPWORDS_SET)}개의 고유한 불용어를 사용합니다. ---")


# --- 2. N-Gram 분석 함수 ---
def analyze_dish_ngrams(csv_path):
    """
    CSV 파일을 읽어 3-gram을 "순수 발견" 방식으로 분석하고,
    '완성된 요리'로 보이는 구문(phrase)의 빈도를 반환 (Counter 객체)
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다.")
        print("스크립트가 루트 폴더에서 실행되고 있는지, 'data/reddit_koreanfood.csv' 경로가 맞는지 확인하세요.")
        return None
    
    df['content'] = df['content'].fillna('')
    df['title'] = df['title'].fillna('')
    all_text = ' '.join(df['title'].str.lower() + ' ' + df['content'].str.lower())
    
    # 3-gram(3단어)만 추출
    trigrams = re.findall(r'\b(\w+ \w+ \w+)\b', all_text)
    
    print(f"--- 3-gram 분석 시작 (총 {len(trigrams)}개 대상) ---")
    
    filtered_phrases = Counter()
    for phrase in trigrams:
        words_in_phrase = set(phrase.split())
        
        # 조건 1: 구문에 불용어가 포함되어 있으면 버림
        if words_in_phrase.intersection(STOPWORDS_SET):
            continue
            
        # 조건 2: 1글자 단어 또는 숫자만 있는 단어는 무시
        if any(len(word) < 2 or word.isdigit() for word in words_in_phrase):
            continue
            
        filtered_phrases[phrase] += 1
            
    print("--- N-gram 분석 완료 ---")
    return filtered_phrases

# --- 3. SQLite DB 생성 함수 (스키마/로직 변경) ---
def create_db_schema(db_path, table_name, data_counter, top_k=15):
    """
    DB 테이블을 생성하고, '메뉴' (랭킹, 이름)을 삽입
    """
    if data_counter is None or not data_counter:
        print("저장할 데이터가 없습니다.")
        # [추가] 데이터가 없을 때도 빈 DB 파일과 테이블은 생성
        print(f"'{db_path}'에 빈 테이블을 생성합니다.")
        data_counter = Counter() # 빈 카운터
    
    # DB 파일이 루트 폴더에 생성되도록 경로 보정
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), db_path)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테이블이 이미 존재하면 삭제
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        print(f"'{table_name}' 테이블이 존재하면 삭제합니다.")
        
        # '레시피' 컬럼들을 NULL을 허용하는 TEXT로 미리 생성
        cursor.execute(f"""
        CREATE TABLE {table_name} (
            ranking INTEGER PRIMARY KEY,
            recipe_name TEXT NOT NULL,
            score INTEGER NOT NULL,
            recipe_detail_ko TEXT, 
            recipe_detail_en TEXT,
            image_url TEXT,
            cook_time INTEGER,
            description TEXT
        )
        """)
        print(f"'{table_name}' 테이블을 새로 생성합니다.")
        
        # 상위 K개의 (이름, 점수) 튜플 리스트
        top_recipes = data_counter.most_common(top_k) 
        
        if top_recipes:
            # (랭킹, 이름, 점수) 데이터만 먼저 삽입
            insert_data = []
            for i, (name, score) in enumerate(top_recipes, 1):
                # (ranking, recipe_name, score)
                insert_data.append((i, name, score))
            
            cursor.executemany(f"INSERT INTO {table_name} (ranking, recipe_name, score) VALUES (?, ?, ?)", insert_data)
            conn.commit()
            
            print(f"\n--- SQLite DB '{db_path}' 생성 완료 ---")
            print(f"'{table_name}' 테이블에 상위 {len(top_recipes)}개 '메뉴' 저장 완료.")
            
            # (검증)
            print("\n--- [DB 검증] 저장된 '메뉴' (Top 5) ---")
            for row in cursor.execute(f"SELECT ranking, recipe_name, score FROM {table_name} ORDER BY ranking ASC LIMIT 5"):
                print(f"- Rank {row[0]}: {row[1]} (Score: {row[2]})")
        
        else:
            print("\n--- SQLite DB '{db_path}' 생성 완료 ---")
            print(f"'{table_name}' 테이블에 저장할 '요리' 키워드를 찾지 못했습니다. (테이블은 비어있음)")
            
        conn.close()

    except sqlite3.Error as e:
        print(f"SQLite 오류 발생: {e}")
    except Exception as e:
        print(f"DB 생성 중 알 수 없는 오류: {e}")


# --- 4. 스크립트 실행 ---
if __name__ == "__main__":
    
    # 스크립트 파일 위치 기준으로 CSV 파일 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_file_path = os.path.join(project_root, CSV_FILE)

    ranking_data = analyze_dish_ngrams(csv_file_path)
    
    # DB 파일을 루트 폴더에 생성
    db_file_path = os.path.join(project_root, DB_FILE)
    create_db_schema(db_file_path, TABLE_NAME, ranking_data, top_k=15)