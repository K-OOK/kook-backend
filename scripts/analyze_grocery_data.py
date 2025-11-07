import pandas as pd
import sqlite3
import os

# --- 1. 설정: 파일 경로 ---

# .py 스크립트가 'scripts' 폴더 안에 있다고 가정하고,
# 루트 폴더에 있는 DB 파일과 data 폴더를 바라보도록 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "grocery_data")

# 데이터를 저장할 DB 파일 (루트 폴더의 kfood_recipes.db)
DB_FILE = os.path.join(BASE_DIR, "kfood_recipes.db")
TABLE_NAME = "grocery_sales" # '상품별 판매량'을 저장할 테이블

def load_dataframes():
    """CSV 파일 7개를 로드"""
    print("--- 7개 CSV 파일 로드를 시작합니다... ---")
    try:
        file_paths = {
            "products": "products.csv",
            "sales": "sales.csv",
            "cities": "cities.csv",
            "categories": "categories.csv",
            "customers": "customers.csv",
            "employees": "employees.csv",
            "countries": "countries.csv"
        }
        
        dataframes = {}
        for name, file in file_paths.items():
            path = os.path.join(DATA_DIR, file)
            dataframes[name] = pd.read_csv(path)
            print(f"✅ '{file}' 로드 성공.")
            
        print("--- 7개 CSV 파일 로드 완료 ---")
        return dataframes

    except FileNotFoundError as e:
        print(f"❌ 오류: {e.filename} 파일을 찾을 수 없습니다.")
        print(f"'{DATA_DIR}' 폴더에 파일이 모두 있는지 확인하세요.")
        return None
    except Exception as e:
        print(f"❌ 파일 로드 중 오류 발생: {e}")
        return None

def analyze_sales(data):
    """데이터를 병합하고 분석 (Cell 2, 3, 4)"""
    
    print("\n--- 분석 시작... ---")
    
    # --- Cell 2: Sales + Products 병합 ---
    merged_df = pd.merge(
        data['sales'], 
        data['products'], 
        on='ProductID', 
        how='left'
    )
    print("✅ (1/3) 'Sales'와 'Products' 병합 완료.")

    # --- Cell 3: 상품별 총 판매량 집계 ---
    product_summary = merged_df.groupby(['ProductID', 'ProductName']).agg(
        TotalQuantity=('Quantity', 'sum'),  # 총 판매량
        TotalRevenue=('TotalPrice', 'sum'),   # 총 매출액
        SalesCount=('SalesID', 'count')     # 총 판매 횟수
    )
    
    # 'TotalQuantity'(총 판매량)을 기준으로 내림차순 정렬
    product_summary_sorted = product_summary.sort_values(
        by='TotalQuantity', 
        ascending=False
    )
    
    # IngredientRank 컬럼 추가 (1부터 시작하는 순위)
    product_summary_sorted['IngredientRank'] = range(1, len(product_summary_sorted) + 1)
    
    print("✅ (2/3) '상품별 판매량' 집계 완료.")
    print("\n--- [인사이트] 상품별 총 판매량 (Quantity) 기준 TOP 10 ---")
    print(product_summary_sorted.head(10))

    # --- Cell 4: 카테고리별 판매 현황 ---
    full_summary_df = pd.merge(
        merged_df, 
        data['categories'], 
        on='CategoryID', 
        how='left'
    )
    
    category_summary = full_summary_df.groupby('CategoryName').agg(
        TotalQuantity=('Quantity', 'sum'),
        TotalRevenue=('TotalPrice', 'sum')
    )
    
    category_summary_sorted = category_summary.sort_values(
        by='TotalQuantity', 
        ascending=False
    )
    
    print("✅ (3/3) '카테고리별 판매량' 집계 완료.")
    print("\n--- [인사이트] 카테고리별 총 판매량 (Quantity) 순위 ---")
    print(category_summary_sorted)
    
    # DB에 저장할 최종 데이터 반환
    return product_summary_sorted

def save_to_db(dataframe, db_path, table_name):
    """분석 결과를 SQLite DB에 저장"""
    
    print(f"\n--- DB 저장 시작 ('{db_path}')... ---")
    try:
        conn = sqlite3.connect(db_path)
        
        # 'ProductID'와 'ProductName'이 groupby 인덱스로 되어있으므로,
        # .reset_index()를 사용해 컬럼으로 풀어줌
        df_to_save = dataframe.reset_index()
        
        # 'GROCERY_SALES' 테이블에 저장. 이미 존재하면 덮어쓰기 (if_exists='replace')
        df_to_save.to_sql(table_name, conn, if_exists='replace', index=False)
        
        print(f"✅ DB 저장 완료! '{table_name}' 테이블이 생성/대체되었습니다.")
        
        # (검증) 저장된 데이터 확인
        print(f"\n--- [DB 검증] '{table_name}' 테이블 상위 5개 ---")
        test_df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(test_df)
        
        conn.close()

    except Exception as e:
        print(f"❌ DB 저장 중 오류 발생: {e}")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 1. 데이터 로드
    all_data = load_dataframes()
    
    if all_data:
        # 2. 데이터 분석
        final_product_sales = analyze_sales(all_data)
        
        # 3. 분석 결과를 DB에 저장
        save_to_db(final_product_sales, DB_FILE, TABLE_NAME)