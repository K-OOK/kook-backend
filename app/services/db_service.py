import sqlite3
from typing import List, Dict, Any
from app.core.config import settings

DB_PATH = settings.DB_PATH

def get_db_connection():
    """DB 연결 객체를 생성 (결과를 dict처럼 사용)"""
    conn = sqlite3.connect(DB_PATH)
    # 쿼리 결과를 '컬럼명'으로 접근할 수 있게 설정 (Pydantic 변환에 필수)
    conn.row_factory = sqlite3.Row 
    return conn

async def get_hot_recipes_from_db(limit: int = 15) -> List[Dict[str, Any]]:
    """
    (Reddit 랭킹) 'hot_recipes' 테이블에서 랜덤으로 4개 메뉴를 조회
    """
    print(f"DB: 'hot_recipes' 테이블에서 랜덤 4개 조회 중...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # ranking, name, score, ko, en, image_url 모두 조회 (랜덤 4개)
        cursor.execute(
            """
            SELECT ranking, recipe_name, image_url, cook_time, description
            FROM hot_recipes 
            ORDER BY RANDOM() 
            LIMIT 4
            """,
        )
        recipes = cursor.fetchall()
        conn.close()
        
        # sqlite3.Row 객체를 Pydantic이 읽을 수 있는 dict 리스트로 반환
        return [dict(row) for row in recipes]
        
    except sqlite3.OperationalError as e:
        print(f"DB 오류: {e}. 'scripts/extract_hot_menus.py'를 실행했는지 확인하세요.")
        return [] # DB나 테이블이 없으면 빈 리스트 반환

async def get_all_recipes_from_db() -> List[Dict[str, Any]]:
    """
    secret API: DB에 저장된 모든 메뉴를 조회
    """
    print(f"DB: 'hot_recipes' 테이블에서 모든 메뉴 조회 중...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT ranking, recipe_name, image_url, cook_time, description, recipe_detail_ko, recipe_detail_en
            FROM hot_recipes
            """
        )
        recipes = cursor.fetchall()
        conn.close()
        return [dict(row) for row in recipes]
    except sqlite3.OperationalError as e:
        print(f"DB 오류: {e}. 'scripts/get_menus_recipes.py'를 실행했는지 확인하세요.")
        return []

async def get_hot_recipes_detail_from_db(ranking: int) -> Dict[str, Any]:
    """
    (기능 2) Hot K-Food 추천 API
    DB(SQLite)에 저장된 메뉴의 디테일을 ranking을 통해 조회
    """
    print(f"DB: 'hot_recipes' 테이블에서 ranking {ranking} 메뉴 조회 중...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT ranking, recipe_name, image_url, cook_time, description, recipe_detail_ko, recipe_detail_en
            FROM hot_recipes 
            WHERE ranking = ?
            """,
            (ranking,)
        )
        recipe = cursor.fetchone()  
        conn.close()
        return dict(recipe)

    except sqlite3.OperationalError as e:
        print(f"DB 오류: {e}. 'scripts/get_menus_recipes.py'를 실행했는지 확인하세요.")
        return {}

async def get_top_ingredients_from_db(limit: int = 10) -> List[Dict[str, Any]]:
    """
    (마트 랭킹) 'grocery_sales' 테이블에서 상위 10개 재료를 조회
    """
    print(f"DB: 'grocery_sales' 테이블에서 Top {limit} 조회 중...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Rank, Name, Quantity 조회
        cursor.execute(
            """
            SELECT IngredientRank, ProductName, TotalQuantity 
            FROM grocery_sales 
            ORDER BY IngredientRank ASC 
            LIMIT 10
            """,
        )
        ingredients = cursor.fetchall()
        conn.close()
        
        # Pydantic 모델 ('TopIngredient')에 맞게 키 이름 변경
        return [
            {
                "ranking": row["IngredientRank"],
                "ingredient_name": row["ProductName"],
                "total_quantity": row["TotalQuantity"]
            } 
            for row in ingredients
        ]
        
    except sqlite3.OperationalError as e:
        print(f"DB 오류: {e}. 'scripts/analyze_grocery_data.py'를 실행했는지 확인하세요.")
        return []