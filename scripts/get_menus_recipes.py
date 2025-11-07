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

# Bedrock 템플릿 (bedrock_service.py에서 복사)
SYSTEM_PROMPT_XML = """
<template>
<recipe>
<title>[ 여기에 요리 제목을 적어주세요 ] (1인분 기준)</title>
<section><title>1. 재료 🥣</title><ingredients>
- [재료 1] ([수량 1])
- [재료 2] ([수량 2])
</ingredients></section>
<section><title>2. 조리 방법 🍳 (총 예상 시간: [총 시간]분)</title><steps>
<step><name>1) [단계 1] (예상 시간: [소요 시간]분)</name><description>
- [상세 설명 1]
- [상세 설명 2]
</description></step>
<step><name>2) [단계 2] (예상 시간: [소요 시간]분)</name><description>
- [상세 설명 1]
</description></step>
</steps></section>
<section><title>3. 곁들여 먹으면 좋은 음료 🥂</title><recommendation>
- [추천 음료 1]
</recommendation></section>
<tip><title>💡 셰프의 꿀팁</title><content>
- [꿀팁 1]
</content></tip>
</recipe>
</template>
"""

SYSTEM_PROMPT_HEADER = """
당신은 "셰프 김"입니다. 사용자의 요청에 맞춰 K-Food 레시피를 생성합니다.
반드시, 반드시 <template> XML 구조를 완벽하게 따라야 합니다.
<template> 태그 바깥에는 어떠한 인사말이나 잡담도 추가하지 마십시오.

<guidelines>
- [규칙 1] 반드시 사용자가 제공한 재료를 활용해야 합니다.
- [규칙 2] "말차 김치", "초콜릿 비빔밥", "민트초코 떡볶이"처럼 맛이 어울리지 않는 터무니없는 레시피는 **절대** 제안해선 안 됩니다.
- [규칙 3] '고추장 버터 불고기', '김치 치즈 파스타', '콘치즈 닭갈비'처럼 (맛이 검증된) 창의적인 퓨전 요리를 우선적으로 제안하세요.
- [규칙 4] 모든 응답은 **한국어**로, 그리고 반드시 아래의 <template> XML 구조를 완벽하게 따라야 합니다.
- [규칙 5] <template> 태그 바깥에는 어떠한 인사말이나 잡담도 추가하지 마십시오.
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

def get_recipe_from_bedrock(menu_name, language="Korean"):
    """Bedrock을 호출하여 XML 레시피를 받아오는 (동기) 함수"""
    
    if language == "English":
        user_query = f"Provide a recipe for {menu_name}."
        system_prompt = f"You are 'Chef Kim'. {SYSTEM_PROMPT_HEADER}\nRespond ONLY in English.\n{SYSTEM_PROMPT_XML}"
    else: # 기본값 (Korean)
        user_query = f"{menu_name} 레시피 알려줘."
        system_prompt = f"You are 'Chef Kim'. {SYSTEM_PROMPT_HEADER}\nRespond ONLY in Korean.\n{SYSTEM_PROMPT_XML}"

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

        # <recipe> 태그만 깔끔하게 추출
        if '<recipe>' in answer:
            answer = "<recipe>" + answer.split('<recipe>', 1)[1]
        if '</recipe>' in answer:
            answer = answer.split('</recipe>', 1)[0] + "</recipe>"
        
        return answer
    
    except Exception as e:
        print(f"  [Bedrock 오류] {menu_name} ({language}): {e}")
        return f"<error>Failed to generate recipe: {e}</error>"

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
    """recipe XML에서 cook_time(총 예상 시간)을 추출하는 함수"""
    try:
        # XML <recipe> 태그 안의 내용만 정확히 추출
        if '<recipe>' in recipe_xml:
            recipe_xml = "<recipe>" + recipe_xml.split('<recipe>', 1)[1]
        if '</recipe>' in recipe_xml:
            recipe_xml = recipe_xml.split('</recipe>', 1)[0] + "</recipe>"
        
        # XML 문자열을 파싱
        root = ET.fromstring(recipe_xml)

        time_minutes = next((m.group(1) for title in root.findall("./section/title") 
                     if title.text and (m := re.search(r'총 예상 시간:\s*(\d+)분', title.text))), None)
        return int(time_minutes) if time_minutes else None
        
        # 한국어 버전: "2. 조리 방법 🍳 (총 예상 시간: 20분)"
        steps_section_title_ko = root.find(".//section/title[starts-with(., '2. 조리 방법 🍳')]")
        if steps_section_title_ko is not None:
            title_text = steps_section_title_ko.text
            match = re.search(r'총 예상 시간:\s*(\d+)', title_text)
            if match:
                return int(match.group(1))
        
        # 영어 버전: "2. Cooking Method 🍳 (Total estimated time: 20 minutes)"
        steps_section_title_en = root.find(".//section/title[starts-with(., '2. Cooking Method 🍳')]")
        if steps_section_title_en is not None:
            title_text = steps_section_title_en.text
            match = re.search(r'Total Time:\s*(\d+)', title_text)
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
        
        # 1. 한글 레시피 가져오기
        print("  - 한글(KO) 레시피 요청 중...")
        recipe_ko = get_recipe_from_bedrock(recipe_name, language="Korean")
        time.sleep(1) # Bedrock API 속도 제한 방지

        # 2. 영어 레시피 가져오기
        print("  - 영어(EN) 레시피 요청 중...")
        recipe_en = get_recipe_from_bedrock(recipe_name, language="English")
        time.sleep(1)

        # 3. Description 가져오기 (영어로 한 줄 설명)
        print("  - Description 요청 중...")
        description = get_description_from_bedrock(recipe_name)
        time.sleep(1)

        # 4. Cook time 추출 (한글 또는 영어 레시피에서)
        cook_time = None
        if recipe_ko and not recipe_ko.startswith("<error>"):
            cook_time = extract_cook_time_from_recipe(recipe_ko)
        if cook_time is None and recipe_en and not recipe_en.startswith("<error>"):
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