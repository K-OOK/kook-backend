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
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
    )
    MODEL_ID = settings.BEDROCK_MODEL_ID
except Exception as e:
    print(f"[Bedrock_Service] Boto3 클라이언트 초기화 실패: {e}")
    bedrock_runtime = None
    MODEL_ID = None

SYSTEM_PROMPT = """
당신은 "셰프 김(Chef Kim)"이라는 이름을 가진, 외국인에게 K-Food를 알려주는 전문 요리사입니다.
당신의 임무는 사용자의 요청에 맞춰, K-Food 레시피를 **한국어**로, 그리고 **매우 명확하고 따라하기 쉬운 형식**으로 제공하는 것입니다.

사용자가 요청할 때, 당신은 반드시, 반드시 아래에 제공된 <template> XML 구조를 완벽하게 따라야 합니다.
<template> 태그 바깥에는 어떠한 인사말이나 잡담도 추가하지 마십시오.

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
</template>
"""
# -----------------------------------------------------------------
# [끝] 템플릿은 여기까지
# -----------------------------------------------------------------

def _parse_recipe_xml_for_preview(xml_string: str) -> Optional[ChatPreviewInfo]:
    """
    Bedrock이 생성한 레시피 XML을 파싱하여 미리보기 정보를 추출
    """
    try:
        # XML <recipe> 태그 안의 내용만 정확히 추출 (이전 코드와 동일)
        if '<recipe>' in xml_string:
            xml_string = "<recipe>" + xml_string.split('<recipe>', 1)[1]
        if '</recipe>' in xml_string:
            xml_string = xml_string.split('</recipe>', 1)[0] + "</recipe>"
            
        # XML 문자열을 파싱
        root = ET.fromstring(xml_string)
        
        # 1. 재료 목록 추출
        # <section> 태그 중 <title>이 "1. 재료 🥣"인 것을 찾음
        ingredients_list = []
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
        # <section> 태그 중 <title>이 "2. 조리 방법 🍳..."으로 시작하는 것을 찾음
        total_time = "정보 없음"
        steps_section_title = root.find(".//section/title[starts-with(., '2. 조리 방법 🍳')]")
        if steps_section_title is not None:
            # title 태그의 텍스트 (예: "2. 조리 방법 🍳 (총 예상 시간: 20분)")
            title_text = steps_section_title.text
            # 정규표현식으로 ( ) 괄호 안의 시간만 추출
            match = re.search(r'\((총 예상 시간:.*?)\)', title_text)
            if match:
                total_time = match.group(1) # "총 예상 시간: 20분"

        return ChatPreviewInfo(
            total_time=total_time,
            ingredients=ingredients_list
        )
        
    except Exception as e:
        print(f"[XML 파싱 오류] {e}")
        # 파싱에 실패해도 미리보기만 못 보낼 뿐, 에러는 아님
        return None

async def generate_recipe_response(user_query: str, ingredients: list = None):
    """
    Bedrock 챗봇을 호출하고, 결과를 파싱하여
    (full_recipe, preview_info) 튜플로 반환
    """
    if not bedrock_runtime:
        error_xml = "<error>Bedrock service is not initialized. AWS credentials를 확인하세요.</error>"
        return ChatResponse(full_recipe=error_xml, preview=None) # 튜플로 반환

    # --- 1. 유저 쿼리와 재료를 합쳐서 'user' 메시지 구성 ---
    if ingredients:
        ingredient_list = ", ".join(ingredients)
        full_query = f"요청 메뉴: {user_query}\n내가 가진 재료: [{ingredient_list}]"
    else:
        full_query = f"요청 메뉴: {user_query}"

    # --- 2. Bedrock API 호출 (Claude 3 모델 기준) ---
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048, # 레시피가 길 수 있으니 넉넉하게
            "system": SYSTEM_PROMPT, # 👈 [중요] 위에서 정의한 시스템 프롬프트
            "messages": [
                {
                    "role": "user",
                    "content": full_query
                }
            ]
        })

        response = bedrock_runtime.invoke_model(...) # (API 호출)

        response_body = json.loads(response.get('body').read())
        full_recipe_xml = response_body.get('content')[0].get('text')
        
        preview_info = _parse_recipe_xml_for_preview(full_recipe_xml)
        
        return ChatResponse(full_recipe=full_recipe_xml, preview=preview_info)

    except Exception as e:
        print(f"[Bedrock_Service] Bedrock API 호출 오류: {e}")
        return f"<error>레시피 생성 중 오류가 발생했습니다: {e}</error>"