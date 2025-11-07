import boto3
import json
from app.core.config import settings
from typing import Optional
import xml.etree.ElementTree as ET
import re
from app.schemas.recipe import ChatPreviewInfo, ChatResponse
# --- [수정 1] Import 추가 ---
from langchain_aws import AmazonKnowledgeBasesRetriever
# ---------------------------

# 설정 파일에서 AWS 정보 로드
try:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.AWS_DEFAULT_REGION,
    )   
    MODEL_ID = settings.BEDROCK_MODEL_ID
    
    # --- [수정 2] Retriever 초기화 로직 추가 ---
    KNOWLEDGE_BASE_ID = settings.KNOWLEDGE_BASE_ID
    if KNOWLEDGE_BASE_ID:
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}},
            region_name=settings.AWS_DEFAULT_REGION,
        )
    else:
        retriever = None
    # -----------------------------------

except Exception as e:
    print(f"[Bedrock_Service] Boto3 클라이언트 또는 Retriever 초기화 실패: {e}")
    bedrock_runtime = None
    retriever = None # 실패 시 retriever도 None
    MODEL_ID = None
    KNOWLEDGE_BASE_ID = None


# --- [수정 3] 'format_docs' 함수 추가 ---
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
# -----------------------------------


def _get_system_prompt(language: str) -> str:
    """
    language에 따라 한국어 또는 영어 시스템 프롬프트를 반환
    (이 함수는 수정 사항 없음)
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
    (이 함수는 수정 사항 없음)
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

# generate_recipe_response 함수만 수정 (다른 함수들은 그대로 둔다고 가정)

async def generate_recipe_response(language: str, ingredients: list = None):
    """
    Bedrock 챗봇을 호출하고, 결과를 파싱하여 ChatResponse 반환
    (KB 사용을 기본으로 전제)
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
    
    # KB 검색 및 기본 쿼리에 사용할 'base_query'
    if ingredients:
        ingredient_list = ", ".join(ingredients)
        if is_english:
            base_query = f"K-Food recipe using these ingredients: [{ingredient_list}]"
            user_query = f"Please create a K-Food recipe using these ingredients: [{ingredient_list}]"
        else:
            base_query = f"재료: [{ingredient_list}]를 사용한 K-Food 레시피"
            user_query = f"내가 가진 재료: [{ingredient_list}]로 K-Food 레시피를 만들어주세요."
    else:
        if is_english:
            base_query = "K-Food recipe"
            user_query = "Please create a K-Food recipe."
        else:
            base_query = "K-Food 레시피"
            user_query = "K-Food 레시피를 만들어주세요."

    # --- 2-1. (수정) KB RAG 로직 (use_kb 파라미터 제거) ---
    context_str = ""
    
    # retriever가 성공적으로 초기화된 경우 (settings.KNOWLEDGE_BASE_ID가 유효한 경우)
    if retriever:
        try:
            print(f"🔍 [KB] Retrieving for query: {base_query}")
            # 참고: retriever.invoke는 동기 함수이므로,
            # 실제 비동기 FastAPI 환경에서는 run_in_threadpool 등을 권장하지만
            # 최소 수정을 위해 일단 동기로 호출합니다.
            retrieved_docs = retriever.invoke(base_query)
            context_str = format_docs(retrieved_docs)
        except Exception as e:
            print(f"⚠️ [KB] Retriever failed: {e}")
            context_str = "Knowledge Base retrieval failed." if is_english else "Knowledge Base 검색에 실패했습니다."
    else:
        # Retriever가 None인 경우 (KB ID가 없거나 초기화 실패)
        print("⚠️ [KB] Retriever is not initialized or KNOWLEDGE_BASE_ID is missing. Skipping KB search.")
        # context_str은 "" (빈 문자열)로 유지됨
    
    # --- 2-2. 최종 쿼리에 KB 컨텍스트 주입 ---
    full_query = ""
    if context_str:
        # KB 검색 결과가 있으면 컨텍스트 주입
        if is_english:
            full_query = f"""Here is some context from the knowledge base. Use this information to create the recipe:
<context>
{context_str}
</context>

User Request: {user_query}
"""
        else:
            full_query = f"""Knowledge Base에서 검색된 참고 자료입니다. 이 정보를 활용해서 레시피를 만들어주세요:
<context>
{context_str}
</context>

사용자 요청: {user_query}
"""
    else:
        # KB를 사용 안 하거나(retriever=None), 검색 결과가 없으면(context_str="") 원래 쿼리 사용
        full_query = user_query 

    # --- 3. Bedrock API 호출 (Claude 3 모델 기준) ---
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,  # 레시피가 길 수 있으니 넉넉하게
            "system": system_prompt,  # 언어에 맞는 시스템 프롬프트
            "messages": [
                {
                    "role": "user",
                    "content": full_query # KB 컨텍스트가 포함되거나 포함되지 않은 최종 쿼리
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