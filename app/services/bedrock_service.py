import boto3
import json
from app.core.config import settings
from typing import Optional, List, Dict, Any
import xml.etree.ElementTree as ET
import re
from app.schemas.recipe import ChatPreviewInfo, ChatResponse
from langchain_aws import AmazonKnowledgeBasesRetriever

# 설정 파일에서 AWS 정보 로드
try:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.AWS_DEFAULT_REGION,
    )   
    MODEL_ID = settings.BEDROCK_MODEL_ID
    
    KNOWLEDGE_BASE_ID = settings.KNOWLEDGE_BASE_ID
    if KNOWLEDGE_BASE_ID:
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}},
            region_name=settings.AWS_DEFAULT_REGION,
        )
    else:
        retriever = None

except Exception as e:
    print(f"[Bedrock_Service] Boto3 클라이언트 또는 Retriever 초기화 실패: {e}")
    bedrock_runtime = None
    retriever = None # 실패 시 retriever도 None
    MODEL_ID = None
    KNOWLEDGE_BASE_ID = None


def format_docs(docs):
    """검색된 문서를 문자열로 변환 (참고 코드에서 가져옴)"""
    if not docs:
        print("⚠️ [KB] 검색된 문서 없음")
        return "" # KB 검색 결과 없으면 빈 문자열 반환

    formatted = []
    for idx, doc in enumerate(docs):
        content = None

        if isinstance(doc, dict):
            # Bedrock KB는 dict 형태로 반환
            content = (
                doc.get("content")
                or doc.get("page_content")
                or doc.get("text")
                or doc.get("excerpt")
            )
            if isinstance(content, dict):
                content = content.get("text") or json.dumps(content, ensure_ascii=False)
        else:
            # LangChain Document 객체
            content = getattr(doc, "page_content", None)

        # 문자열 변환
        if content:
            if isinstance(content, str):
                content_str = content.strip()
            else:
                content_str = str(content)

            if content_str:
                formatted.append(content_str)

    result = (
        "\n\n---\n\n".join(formatted)
        if formatted
        else "" # 문서 내용은 있으나 추출 실패 시 빈 문자열
    )
    print(f"✅ [KB] {len(formatted)}개 문서 포맷 완료 (총 {len(result)}자)")
    return result


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
- [Rule 1] **[Mandatory] Ingredient Utilization:** You MUST prioritize using the ingredients provided by the user.

- [Rule 2] **[Critical] Taste Validation & KB Usage:**
  1. **(Forbidden)** NEVER suggest absurd, unpalatable combinations (e.g., "Matcha Kimchi", "Chocolate Bibimbap", "Mint Chocolate Tteokbokki").
  2. **(Required)** You MUST consult the Knowledge Base (KB) to provide a validated recipe.
  3. **(Fallback)** If the KB has no validated recipe for the user's ingredients, OR the only possible combination is absurd (see #1), you MUST suggest an **alternative K-Food dish** that uses **similar ingredients**. (e.g., Instead of 'Mint Chocolate' and 'Tteokbokki', suggest a 'Choco Rice Cake Pie' using 'Chocolate' and 'Rice Cakes').

- [Rule 3] **[Priority] Proven Fusion:** Prioritize creative but validated fusion dishes (e.g., 'Gochujang Butter Bulgogi', 'Kimchi Cheese Pasta', 'Corn Cheese Dakgalbi').

- [Rule 4] **[Format] Output:** The response MUST be in **English** and MUST strictly adhere to the provided `<template>` XML structure.

- [Rule 5] **[Constraint] No Chatter:** DO NOT add any text (greetings, explanations, etc.) outside the `<template>` tags.

- [Rule 6] **[Format-Ingredients] Ingredient Format:** All ingredients in the <ingredients> section MUST strictly follow the "Ingredient Name (Quantity)" format. (e.g., Sesame oil (1 tablespoon))
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
        return """당신은 "셰프 김(Chef Kim)"이라는 이름을 가진, 외국인에게 **K-Food(한식)**를 알려주는 전문 요리사입니다.
당신의 임무는 사용자의 요청에 맞춰, K-Food 레시피를 **한국어**로, 그리고 **매우 명확하고 따라하기 쉬운 형식**으로 제공하는 것입니다.
모든 레시피는 반드시 한식 또는 퓨전 한식의 범위 안에서 추천되어야 합니다. 한식의 특성에 어긋나는 경우, 가이드라인에 따라 다른 대안을 제시해야 합니다.

사용자가 요청할 때, 당신은 반드시, 반드시 아래에 제공된 <template> XML 구조를 완벽하게 따라야 합니다.
<template> 태그 바깥에는 어떠한 인사말이나 잡담도 추가하지 마십시오.

<guidelines>
- [규칙 1] **[Mandatory] 재료 활용:** 사용자가 명시한 재료를 **최우선**으로 활용해야 합니다.

- [규칙 2] **[Critical] 맛 검증 및 KB 활용:** 1. **(금지)** "말차 김치", "초콜릿 비빔밥", "민트초코 떡볶이"처럼 맛이 어울리지 않는 터무니없는 조합은 **절대** 제안하지 않습니다.
  3. **(대안 제시)** 만약 KB에 사용자의 재료로 만들 수 있는 검증된 레시피가 없거나, 유일한 조합이 (1)에서 금지한 터무니없는 레시피일 경우, 원본 재료와 **유사한 재료**를 사용하는 **다른 한식 레시피**를 대안으로 추천하세요. (예: '민트초코'와 '떡볶이' 대신, '초콜릿'과 '떡'을 활용한 '초코 찰떡 파이'를 제안) 대안을 제안할 때도 <template>형식을 반드시 따라야 합니다.

- [규칙 3] **[Priority] 검증된 퓨전:** '고추장 버터 불고기', '김치 치즈 파스타', '콘치즈 닭갈비'처럼 (맛이 검증된) 창의적인 퓨전 요리를 **우선적으로** 제안하세요.

- [규칙 4] **[Format] 출력 형식:** 응답은 **반드시 한국어**로, 제공된 `<template>` XML 구조를 완벽하게 준수해야 합니다.

- [규칙 5] **[Constraint] 잡담 금지:** `<template>` 태그 외부에 어떤 텍스트(인사, 설명 등)도 추가하지 마십시오.

- [규칙 6] **[Format-Ingredients] 재료 형식:** <ingredients> 섹션의 모든 재료는 "재료명 (수량)" 형식을 엄격하게 따라야 합니다. (예: 간장 (2큰술))
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

def create_bedrock_payload(
    language: str,
    ingredients: List[str],
    chat_history: List[Dict[str, str]], 
    context_str: str
) -> Dict[str, Any]:
    """
    Bedrock API 호출에 필요한 최종 JSON Payload를 생성하여 반환 (스트리밍용)
    """
    is_english = language.lower() == "eng"
    system_prompt = _get_system_prompt(language)
    
    # 1. base_query 및 user_message 구성
    if ingredients:
        ingredient_list = ", ".join(ingredients)
        base_query = f"K-Food recipe using: {ingredient_list}" if is_english else f"재료: {ingredient_list} K-Food 레시피"
    else:
        # 이 경우는 첫 질문이거나 꼬리 질문이 재료 없이 들어온 경우 (단순 추천)
        base_query = "K-Food recipe" if is_english else "K-Food 레시피"

    is_first_message = not chat_history 

    # 2. 사용자 메시지 구성
    if is_first_message and context_str:
        # KB 컨텍스트 주입 (첫 질문)
        user_message = f"""Here is some context. Use this to create the recipe:
<context>{context_str}</context>
User Request: {base_query}""" if is_english else f"""KB 참고 자료입니다:
<context>{context_str}</context>
사용자 요청: {base_query}"""
    else:
        # 꼬리 질문 시나리오: ingredients 리스트의 첫 번째 요소를 꼬리 질문 텍스트로 간주
        # (router에서 payload.ingredients[0]에 실제 꼬리 질문 텍스트를 담아 보낸다고 가정)
        if not is_first_message and ingredients and len(ingredients) > 0:
            user_message = ingredients[0]
        else:
            user_message = base_query


    # 3. 메시지 히스토리 정리 및 추가
    messages = []
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    # 최종 Payload
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": messages,
        "stream": True,
        "model_id": MODEL_ID # Model ID를 페이로드에 포함하여 router에서 사용
    }

    return body