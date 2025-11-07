import sqlite3
import random
from app.core.config import settings # config.py에서 DB 경로 가져오기

DB_PATH = settings.DB_PATH
TABLE_NAME = "hot_recipes"

def get_db_connection():
    """DB 연결 객체 생성 (결과를 딕셔너리처럼 사용)"""
    conn = sqlite3.connect(DB_PATH)
    # 쿼리 결과를 '컬럼명'으로 접근할 수 있게 설정
    conn.row_factory = sqlite3.Row 
    return conn

async def get_hot_recipes_from_db(k: int = 15):
    """DB에서 상위 K개 레시피를 가져옴"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # score가 높은 순(DESC)으로 K개 가져오기
        cursor.execute(f"SELECT recipe_name, score FROM {TABLE_NAME} ORDER BY score DESC LIMIT ?", (k,))
        recipes = cursor.fetchall()
        conn.close()
        
        # sqlite3.Row 객체를 dict로 변환 (JSON으로 만들기 쉽게)
        return [dict(row) for row in recipes]
        
    except sqlite3.OperationalError as e:
        print(f"DB 오류: {e}. 'scripts/seed_sqlite.py'를 실행했는지 확인하세요.")
        return [] # DB나 테이블이 없으면 빈 리스트 반환

async def get_random_recommendations(total: int = 15, sample_size: int = 4):
    """
    DB에서 상위 'total'개를 가져온 뒤, 'sample_size'만큼 랜덤으로 추천
    """
    top_recipes = await get_hot_recipes_from_db(k=total)
    
    if not top_recipes:
        return ["Kimchi Jjigae", "Bulgogi", "Bibimbap"] # DB가 비어있을 때 대비

    # 레시피 이름만 추출
    recipe_names = [recipe['recipe_name'] for recipe in top_recipes]
    
    # 만약 15개보다 적으면 그냥 다 반환
    if len(recipe_names) <= sample_size:
        return recipe_names
        
    # 15개 중에서 3~4개 랜덤 샘플링
    return random.sample(recipe_names, sample_size)