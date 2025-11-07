import boto3
import json
from app.core.config import settings
from typing import Optional
import xml.etree.ElementTree as ET
import re
from app.schemas.recipe import ChatPreviewInfo, ChatResponse

# 설정 파일에서 AWS 정보 로드
try:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.AWS_DEFAULT_REGION,
    )   
    MODEL_ID = settings.BEDROCK_MODEL_ID
except Exception as e:
    print(f"[Bedrock_Service] Boto3 클라이언트 초기화 실패: {e}")
    bedrock_runtime = None
    MODEL_ID = None

def _get_system_prompt(language: str) -> str:
    """
    language에 따라 한국어 또는 영어 시스템 프롬프트를 반환
    """
    if language.lower() == "eng":
        return """You are "Chef Kim", a professional chef who introduces K-Food to foreigners.
Your mission is to provide K-Food recipes in **English** in a **very clear and easy-to-follow format** based on user requests.

When users make requests, you must strictly follow the <template> XML structure provided below.
Do not add any greetings or small talk outside the <template> tags.

<guidelines>
- [Rule 1] You must use the ingredients provided by the user.
- [Rule 2] You must **never** suggest absurd recipes that don't taste good together, like "matcha kimchi", "chocolate bibimbap", or "mint chocolate tteokbokki".
- [Rule 3] Prioritize creative fusion dishes with proven flavors like 'gochujang butter bulgogi', 'kimchi cheese pasta', or 'corn cheese dakgalbi'.
- [Rule 4] All responses must be in **English** and must strictly follow the <template> XML structure below.
- [Rule 5] Do not add any greetings or small talk outside the <template> tags.
</guidelines>

<template>
<recipe>

<title>
[ Write the dish title here ] (for 1 serving)
</title>

<section>
<title>1. Ingredients 🥣</title>
<ingredients>
- [Ingredient 1] ([Quantity 1, e.g., 100g or 1 tablespoon])
- [Ingredient 2] ([Quantity 2])
- (List all ingredients in this format)
</ingredients>
</section>

<section>
<title>2. Cooking Method 🍳 (Total estimated time: [total time] minutes)</title>
<steps>
<step>
<name>1) [Step 1 name, e.g., Prepare ingredients] (Estimated time: [time] minutes)</name>
<description>
- [Detailed description 1 for this step]
- [Detailed description 2 for this step]
</description>
</step>
<step>
<name>2) [Step 2 name, e.g., Stir-fry vegetables] (Estimated time: [time] minutes)</name>
<description>
- [Detailed description 1 for this step]
- [Detailed description 2 for this step]
</description>
</step>
<step>
<name>3) [Step 3 name, e.g., Add sauce and simmer] (Estimated time: [time] minutes)</name>
<description>
- [Detailed description 1 for this step]
</description>
</step>
</steps>
</section>

<section>
<title>3. Recommended Drinks 🥂</title>
<recommendation>
- [Recommended drink 1, e.g., makgeolli or beer]
</recommendation>
</section>

<tip>
<title>💡 Chef's Tip</title>
<content>
- [Tip 1 to make this dish easier or more delicious]
- [Interesting fact about this dish (optional)]
</content>
</tip>

</recipe>
</template>"""
    else:  # 한국어 (기본값)
        return """당신은 "셰프 김(Chef Kim)"이라는 이름을 가진, 외국인에게 K-Food를 알려주는 전문 요리사입니다.
당신의 임무는 사용자의 요청에 맞춰, K-Food 레시피를 **한국어**로, 그리고 **매우 명확하고 따라하기 쉬운 형식**으로 제공하는 것입니다.

사용자가 요청할 때, 당신은 반드시, 반드시 아래에 제공된 <template> XML 구조를 완벽하게 따라야 합니다.
<template> 태그 바깥에는 어떠한 인사말이나 잡담도 추가하지 마십시오.

<guidelines>
- [규칙 1] 반드시 사용자가 제공한 재료를 활용해야 합니다.
- [규칙 2] "말차 김치", "초콜릿 비빔밥", "민트초코 떡볶이"처럼 맛이 어울리지 않는 터무니없는 레시피는 **절대** 제안해선 안 됩니다.
- [규칙 3] '고추장 버터 불고기', '김치 치즈 파스타', '콘치즈 닭갈비'처럼 (맛이 검증된) 창의적인 퓨전 요리를 우선적으로 제안하세요.
- [규칙 4] 모든 응답은 **한국어**로, 그리고 반드시 아래의 <template> XML 구조를 완벽하게 따라야 합니다.
- [규칙 5] <template> 태그 바깥에는 어떠한 인사말이나 잡담도 추가하지 마십시오.
</guidelines>

<template>
<recipe>

<title>
[ 여기에 요리 제목을 적어주세요 ] (1인분 기준)
</title>

<section>
<title>1. 재료 🥣</title>
<ingredients>
- [재료 1] ([수량 1, 예: 100g 또는 1큰술])
- [재료 2] ([수량 2])
- (모든 재료를 이 형식으로 나열)
</ingredients>
</section>

<section>
<title>2. 조리 방법 🍳 (총 예상 시간: [총 시간]분)</title>
<steps>
<step>
<name>1) [단계 1 이름, 예: 재료 준비하기] (예상 시간: [소요 시간]분)</name>
<description>
- [이 단계의 상세한 설명 1]
- [이 단계의 상세한 설명 2]
</description>
</step>
<step>
<name>2) [단계 2 이름, 예: 야채 볶기] (예상 시간: [소요 시간]분)</name>
<description>
- [이 단계의 상세한 설명 1]
- [이 단계의 상세한 설명 2]
</description>
</step>
<step>
<name>3) [단계 3 이름, 예: 소스 넣고 끓이기] (예상 시간: [소요 시간]분)</name>
<description>
- [이 단계의 상세한 설명 1]
</description>
</step>
</steps>
</section>

<section>
<title>3. 곁들여 먹으면 좋은 음료 🥂</title>
<recommendation>
- [추천 음료 1, 예: 막걸리 또는 맥주]
</recommendation>
</section>

<tip>
<title>💡 셰프의 꿀팁</title>
<content>
- [이 요리를 더 쉽게 하거나 맛있게 만드는 비법 1]
- [이 요리와 관련된 재미있는 사실 (선택 사항)]
</content>
</tip>

</recipe>
</template>"""

def _parse_recipe_xml_for_preview(xml_string: str, language: str = "kor") -> Optional[ChatPreviewInfo]:
    """
    Bedrock이 생성한 레시피 XML을 파싱하여 미리보기 정보를 추출
    language에 따라 한국어/영어 버전을 지원
    """
    try:
        # XML <recipe> 태그 안의 내용만 정확히 추출
        if '<recipe>' in xml_string:
            xml_string = "<recipe>" + xml_string.split('<recipe>', 1)[1]
        if '</recipe>' in xml_string:
            xml_string = xml_string.split('</recipe>', 1)[0] + "</recipe>"
            
        # XML 문자열을 파싱
        root = ET.fromstring(xml_string)
        
        # 언어에 따라 다른 키워드 사용
        is_english = language.lower() == "eng"
        
        # 1. 재료 목록 추출
        ingredients_list = []
        if is_english:
            ingredients_section = root.find(".//section[title='1. Ingredients 🥣']")
        else:
            ingredients_section = root.find(".//section[title='1. 재료 🥣']")
        
        if ingredients_section is not None:
            ingredients_tag = ingredients_section.find('ingredients')
            if ingredients_tag is not None:
                # ingredients 태그의 텍스트를 줄바꿈 기준으로 분리
                ingredients_list = [
                    line.strip() for line in ingredients_tag.text.strip().split('\n') 
                    if line.strip()
                ]

        # 2. 총 조리 시간 추출
        total_time = "정보 없음" if not is_english else "Information not available"
        if is_english:
            steps_section_title = root.find(".//section/title[starts-with(., '2. Cooking Method 🍳')]")
            if steps_section_title is not None:
                title_text = steps_section_title.text
                match = re.search(r'\((Total estimated time:.*?)\)', title_text)
                if match:
                    total_time = match.group(1)  # "Total estimated time: 20 minutes"
        else:
            steps_section_title = root.find(".//section/title[starts-with(., '2. 조리 방법 🍳')]")
            if steps_section_title is not None:
                title_text = steps_section_title.text
                match = re.search(r'\((총 예상 시간:.*?)\)', title_text)
                if match:
                    total_time = match.group(1)  # "총 예상 시간: 20분"

        return ChatPreviewInfo(
            total_time=total_time,
            ingredients=ingredients_list
        )
        
    except Exception as e:
        print(f"[XML 파싱 오류] {e}")
        # 파싱에 실패해도 미리보기만 못 보낼 뿐, 에러는 아님
        return None

async def generate_recipe_response(language: str, ingredients: list = None):
    """
    Bedrock 챗봇을 호출하고, 결과를 파싱하여 ChatResponse 반환
    language: "kor" (한국어) 또는 "eng" (영어)
    """
    if not bedrock_runtime:
        error_msg = "Bedrock service is not initialized. Please check AWS credentials."
        if language.lower() != "eng":
            error_msg = "Bedrock service가 초기화되지 않았습니다. AWS credentials를 확인하세요."
        error_xml = f"<error>{error_msg}</error>"
        return ChatResponse(full_recipe=error_xml, preview=None)

    # --- 1. 언어에 맞는 시스템 프롬프트 가져오기 ---
    system_prompt = _get_system_prompt(language)
    
    # --- 2. 유저 쿼리와 재료를 합쳐서 'user' 메시지 구성 ---
    is_english = language.lower() == "eng"
    if ingredients:
        ingredient_list = ", ".join(ingredients)
        if is_english:
            full_query = f"Please create a K-Food recipe using these ingredients: [{ingredient_list}]"
        else:
            full_query = f"내가 가진 재료: [{ingredient_list}]로 K-Food 레시피를 만들어주세요."
    else:
        if is_english:
            full_query = "Please create a K-Food recipe."
        else:
            full_query = "K-Food 레시피를 만들어주세요."

    # --- 3. Bedrock API 호출 (Claude 3 모델 기준) ---
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,  # 레시피가 길 수 있으니 넉넉하게
            "system": system_prompt,  # 언어에 맞는 시스템 프롬프트
            "messages": [
                {
                    "role": "user",
                    "content": full_query
                }
            ]
        })

        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=body
        )

        response_body = json.loads(response.get('body').read())
        full_recipe_xml = response_body.get('content')[0].get('text')
        
        preview_info = _parse_recipe_xml_for_preview(full_recipe_xml, language)
        
        return ChatResponse(full_recipe=full_recipe_xml, preview=preview_info)

    except Exception as e:
        print(f"[Bedrock_Service] Bedrock API 호출 오류: {e}")
        error_msg = f"An error occurred while generating the recipe: {e}"
        if language.lower() != "eng":
            error_msg = f"레시피 생성 중 오류가 발생했습니다: {e}"
        return ChatResponse(full_recipe=f"<error>{error_msg}</error>", preview=None)