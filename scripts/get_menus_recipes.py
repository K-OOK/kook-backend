# scripts/get_menus_recipes.py

import sqlite3
import boto3
import json
import time
import sys
import xml.etree.ElementTree as ET
import re

# 이 스크립트는 app 모듈(config)을 사용하므로,
# 'python -m scripts.get_menus_recipes'로 실행해야 함
try:
    from app.core.config import settings
except ModuleNotFoundError:
    print("---------------------------------------------------------------")
    print("오류: 이 스크립트는 모듈로 실행해야 합니다.")
    print("프로젝트 루트(kook_backend) 폴더에서")
    print("\n  python -m scripts.get_menus_recipes\n")
    print("---------------------------------------------------------------")
    sys.exit(1)


# --- 1. 설정 ---
DB_FILE = settings.DB_PATH # config에서 DB 경로 가져오기
TABLE_NAME = 'hot_recipes'

# Bedrock 템플릿 (영어 메인)
SYSTEM_PROMPT_XML = """
<template>
<recipe>
<title>[ Write the dish title here ] (for 1 serving)</title>
<section><title>1. Ingredients 🥣</title><ingredients>
- [Ingredient 1] ([Quantity 1, e.g., 100g or 1 tablespoon])
- [Ingredient 2] ([Quantity 2])
</ingredients></section>
<section><title>2. Cooking Method 🍳 (Total estimated time: [total time] minutes)</title><steps>
<step><name>1) [Step 1 name, e.g., Prepare ingredients] (Estimated time: [time] minutes)</name><description>
- [Detailed description 1 for this step]
- [Detailed description 2 for this step]
</description></step>
<step><name>2) [Step 2 name, e.g., Stir-fry vegetables] (Estimated time: [time] minutes)</name><description>
- [Detailed description 1 for this step]
- [Detailed description 2 for this step]
</description></step>
<step><name>3) [Step 3 name, e.g., Add sauce and simmer] (Estimated time: [time] minutes)</name><description>
- [Detailed description 1 for this step]
</description></step>
</steps></section>
<section><title>3. Recommended Drinks 🥂</title><recommendation>
- [Recommended drink 1, e.g., makgeolli or beer]
</recommendation></section>
<tip><title>💡 Chef's Tip</title><content>
- [Tip 1 to make this dish easier or more delicious]
- [Interesting fact about this dish (optional)]
</content></tip>
</recipe>
</template>
"""

SYSTEM_PROMPT_HEADER = """You are "Chef Kim", a professional chef who introduces K-Food to foreigners.
Your mission is to provide K-Food recipes in **English** in a **very clear and easy-to-follow format** based on user requests.

<guidelines>
1.  **Ingredient Usage (MANDATORY):** You MUST utilize the ingredients provided by the user in your recipe suggestions.
2.  **Taste Validation (STRICTLY FORBIDDEN):** NEVER suggest absurd or unpalatable combinations (e.g., "Matcha Kimchi", "Chocolate Bibimbap", "Mint Chocolate Tteokbokki"). All recipes must be culinarily sound.
3.  **Prioritize Proven Fusion:** Focus on creative but validated flavor profiles.
    * *Good Examples:* Gochujang Butter Bulgogi, Kimchi Cheese Pasta, Corn Cheese Dakgalbi.
4.  **Language:** All responses must be in **ENGLISH**.
5.  **Output Format:**
    * Your entire response must be strictly contained within the <template> XML structure provided below.
    * DO NOT include any introductory text, greetings, or small talk outside the XML tags.
</guidelines>
"""

# --- 2. Bedrock 클라이언트 및 헬퍼 함수 ---
try:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.AWS_DEFAULT_REGION,
    )
    MODEL_ID = settings.BEDROCK_MODEL_ID
except Exception as e:
    print(f"Boto3 클라이언트 초기화 실패: {e}")
    sys.exit(1)

def _extract_recipe_xml(text):
    """응답 텍스트에서 <recipe> 태그만 깔끔하게 추출하는 헬퍼 함수"""
    if '<recipe>' in text:
        text = "<recipe>" + text.split('<recipe>', 1)[1]
    if '</recipe>' in text:
        text = text.split('</recipe>', 1)[0] + "</recipe>"
    return text

def get_recipe_from_bedrock(menu_name):
    """Bedrock을 호출하여 영어 XML 레시피를 받아오는 함수"""
    user_query = f"Provide a recipe for {menu_name}."
    system_prompt = f"{SYSTEM_PROMPT_HEADER}\n{SYSTEM_PROMPT_XML}"

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_query}]
        })

        response = bedrock_runtime.invoke_model(
            body=body, modelId=MODEL_ID, contentType='application/json', accept='application/json'
        )
        response_body = json.loads(response.get('body').read())
        answer = response_body.get('content')[0].get('text')
        
        return _extract_recipe_xml(answer)
    
    except Exception as e:
        print(f"  [Bedrock 오류] {menu_name}: {e}")
        return f"<error>Failed to generate recipe: {e}</error>"

def translate_recipe_to_korean(recipe_xml_en):
    """영어 레시피 XML을 한글로 번역하는 함수"""
    system_prompt = """You are a professional translator specializing in Korean food recipes.
Your task is to translate the provided English recipe XML into Korean, maintaining the exact same XML structure and format.
Translate all content including titles, ingredients, steps, recommendations, and tips.
Keep the XML tags and structure exactly the same - only translate the text content.
Do not add any greetings or extra text. Just provide the translated XML directly."""

    user_query = f"Translate the following recipe XML from English to Korean, maintaining the exact XML structure:\n\n{recipe_xml_en}"

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_query}]
        })

        response = bedrock_runtime.invoke_model(
            body=body, modelId=MODEL_ID, contentType='application/json', accept='application/json'
        )
        response_body = json.loads(response.get('body').read())
        translated_xml = response_body.get('content')[0].get('text').strip()
        
        return _extract_recipe_xml(translated_xml)
    
    except Exception as e:
        print(f"  [Bedrock 오류] Translation: {e}")
        return None

def get_description_from_bedrock(menu_name):
    """Bedrock을 호출하여 음식에 대한 한 줄 설명(영어)을 받아오는 함수"""
    
    system_prompt = """You are a food expert. Provide a concise one-sentence description of Korean dishes in English.
The description should be clear and informative, explaining what the dish is made of.
Example: For "김밥" (kimbap), you would say: "Seaweed-wrapped rice rolls filled with vegetables, egg, and sometimes meat or tuna."
Do not include any greetings or extra text. Just provide the description directly."""

    user_query = f"Provide a one-sentence description for {menu_name}."

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,  # 한 줄 설명이므로 짧게
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_query}]
        })

        response = bedrock_runtime.invoke_model(
            body=body, modelId=MODEL_ID, contentType='application/json', accept='application/json'
        )
        response_body = json.loads(response.get('body').read())
        description = response_body.get('content')[0].get('text').strip()
        
        return description
    
    except Exception as e:
        print(f"  [Bedrock 오류] Description for {menu_name}: {e}")
        return None

def extract_cook_time_from_recipe(recipe_xml):
    """영어 recipe XML에서 cook_time(Total estimated time)을 추출하는 함수"""
    try:
        root = ET.fromstring(_extract_recipe_xml(recipe_xml))
        
        for title in root.findall("./section/title"):
            if title.text:
                match = re.search(r'Total estimated time:\s*(\d+)\s*minutes?', title.text) or \
                       re.search(r'Total Time:\s*(\d+)\s*minutes?', title.text)
                if match:
                    return int(match.group(1))

        return None
    
    except Exception as e:
        print(f"  [Cook time 추출 오류] {e}")
        return None

# --- 3. 메인 실행 로직 ---
def enrich_database():
    print(f"'{DB_FILE}'의 'hot_recipes' 테이블 레시피 자동 채우기를 시작합니다.")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # '할 일 목록' (레시피가 비어있는 항목) 가져오기
    cursor.execute("SELECT ranking, recipe_name FROM hot_recipes WHERE recipe_detail_ko IS NULL")
    tasks = cursor.fetchall()
    
    if not tasks:
        print("모든 레시피가 이미 채워져 있습니다. 작업을 종료합니다.")
        conn.close()
        return

    print(f"총 {len(tasks)}개의 메뉴에 대한 레시피를 Bedrock에서 가져옵니다...")

    for ranking, recipe_name in tasks:
        print(f"\n[작업 {ranking}/{len(tasks)}] '{recipe_name}' 레시피 가져오는 중...")
        
        # 1. 영어 레시피 가져오기 (메인)
        print("  - 영어(EN) 레시피 요청 중...")
        recipe_en = get_recipe_from_bedrock(recipe_name)
        time.sleep(1) # Bedrock API 속도 제한 방지
        
        if recipe_en and not recipe_en.startswith("<error>"):
            # 2. 영어 레시피를 한글로 번역
            print("  - 한글(KO) 번역 중...")
            recipe_ko = translate_recipe_to_korean(recipe_en)
            time.sleep(1)
            
            # 번역 실패 시 None으로 설정
            if not recipe_ko or recipe_ko.startswith("<error>"):
                recipe_ko = None
                print("  ⚠️  번역 실패 - 한글 레시피를 건너뜁니다.")
        else:
            recipe_ko = None
            print("  ⚠️  영어 레시피 생성 실패 - 한글 레시피를 건너뜁니다.")

        # 3. Description 가져오기 (영어로 한 줄 설명)
        print("  - Description 요청 중...")
        description = get_description_from_bedrock(recipe_name)
        time.sleep(1)

        # 4. Cook time 추출 (영어 레시피에서)
        cook_time = None
        if recipe_en and not recipe_en.startswith("<error>"):
            cook_time = extract_cook_time_from_recipe(recipe_en)

        # 5. DB에 업데이트
        cursor.execute(
            f"UPDATE {TABLE_NAME} SET recipe_detail_ko = ?, recipe_detail_en = ?, cook_time = ?, description = ? WHERE ranking = ?",
            (recipe_ko, recipe_en, cook_time, description, ranking)
        )
        conn.commit()
        print(f"  ✅ '{recipe_name}' (Rank {ranking}) DB 업데이트 완료.")
        if cook_time:
            print(f"     Cook time: {cook_time}분")
        if description:
            print(f"     Description: {description}")

    conn.close()
    print("\n--- 모든 레시피 자동 채우기 작업 완료! ---")

# --- 4. 스크립트 실행 ---
if __name__ == "__main__":
    enrich_database()