# app/services/bedrock_service.py (LangChain 기반 최종 수정)

import boto3
import json

from langchain_core.language_models import LLM
from app.core.config import settings
from typing import Optional, List, Dict, Any
import xml.etree.ElementTree as ET
import re
from app.schemas.recipe import ChatPreviewInfo, ChatResponse

# --- [수정 1] Boto3 대신 LangChain 객체 임포트 (기존 코드 유지) ---
from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence # 체인 타입
from langchain_core.messages import HumanMessage, AIMessage # 메시지 타입
# ---------------------------

# 설정 파일에서 AWS 정보 로드 (기존 코드 유지)
try:
    bedrock_runtime = None
    llm = None
    MODEL_ID = settings.BEDROCK_MODEL_ID
    
    # 🔴 [retriever]만 LangChain 객체로 유지 (토큰은 요청 시 갱신됨)
    KNOWLEDGE_BASE_ID = settings.KNOWLEDGE_BASE_ID

except Exception as e:
    print(f"[Bedrock_Service] LangChain LLM 또는 Retriever 초기화 실패: {e}")
    bedrock_runtime = None
    llm = None
    retriever = None 
    MODEL_ID = None
    KNOWLEDGE_BASE_ID = None

# 토큰 만료 방지를 위한 함수
def get_fresh_llm(region: str, model_id: str):
    """요청 시마다 새로운 LLM 객체를 생성하여 토큰 만료를 방지"""
    return ChatBedrock(
        model_id=model_id,
        region_name=region,
        model_kwargs={
            "max_tokens": 4096, 
            "temperature": 0.2, 
            "top_p": 0.6
        },
        streaming=True,
    )

# 위의 llm과 비슷하게 토큰 만료 방지 위한 함수
def get_fresh_retriever():
    """요청 시마다 새로운 Retriever 객체를 생성하여 토큰 만료를 방지"""
    if not KNOWLEDGE_BASE_ID:
        return None
    return AmazonKnowledgeBasesRetriever(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}},
        region_name=settings.AWS_DEFAULT_REGION,
    )

def format_docs(docs):
    """KB 검색된 문서를 문자열로 변환하여 RAG 컨텍스트로 사용."""
    if not docs:
        print("⚠️ [KB] 검색된 문서 없음")
        return ""
    formatted = []
    for idx, doc in enumerate(docs):
        content = None
        if isinstance(doc, dict):
            content = (
                doc.get("content")
                or doc.get("page_content")
                or doc.get("text")
                or doc.get("excerpt")
            )
            if isinstance(content, dict):
                content = content.get("text") or json.dumps(content, ensure_ascii=False)
        else:
            content = getattr(doc, "page_content", None)
        if content:
            if isinstance(content, str):
                content_str = content.strip()
            else:
                content_str = str(content)
            if content_str:
                formatted.append(content_str)
    result = ("\n\n---\n\n".join(formatted) if formatted else "")
    print(f"✅ [KB] {len(formatted)}개 문서 포맷 완료 (총 {len(result)}자)")
    return result


def _get_system_prompt(language: str) -> str:
    """
    language에 따라 한국어 또는 영어 시스템 프롬프트를 반환
    """
    if language.lower() == "eng":
        return """You are "Chef Kim", a professional chef who introduces **K-Food (which means Hansik, or Korean Cuisine)** to foreigners.
Your mission is to provide K-Food recipes in **English** in a **very clear and easy-to-follow format** based on user requests.

When users make requests, you must strictly follow the <template> XML structure provided below.
Do not add any greetings or small talk outside the <template> tags.

<guidelines>
- [Rule 0] **[Core Identity] K-Food = Hansik:** "K-Food" means "Hansik" (Korean cuisine). Your **core mission** is to recommend **only Hansik** or **Fusion Hansik** recipes. If a request falls outside the scope of Hansik (in terms of taste, ingredients, or methods), you must apply the fallback principle from [Rule 2.3] and suggest a Hansik-based alternative.

- [Rule 1] **[Mandatory] Ingredient Utilization:** You MUST prioritize using the ingredients provided by the user.

- [Rule 2] **[Critical] Taste Validation & KB Usage:**
  1. **(Forbidden)** NEVER suggest absurd, unpalatable combinations (e.g., "Matcha Kimchi", "Chocolate Bibimbap", "Mint Chocolate Tteokbokki").
  2. **(Required)** You MUST consult the Knowledge Base (KB) to provide a validated recipe.
  3. **(Fallback)** If the KB has no validated recipe for the user's ingredients, OR the only possible combination is absurd (see #1), you MUST suggest an **alternative K-Food dish** that uses **similar ingredients**. (e.g., Instead of 'Mint Chocolate' and 'Tteokbokki', suggest a 'Choco Rice Cake Pie' using 'Chocolate' and 'Rice Cakes').

- [Rule 3] **[Priority] Focus on Stability:** To prevent absurd recommendations, propose only conservative, flavor-verified Hansik-based fusion menus (e.g., Cheese Dakgalbi, Cheese Fried Rice). Focus on stability rather than excessive creativity.

- [Rule 4] **[Audience] Target: Americans & Ingredient Restriction (CRITICAL):** All recipes must be suitable for a standard American kitchen. Prioritize ingredients that are **easily accessible in major US supermarkets** (e.g., Kroger, Walmart, Costco). **Specifically, ABSOLUTELY AVOID using difficult-to-find traditional Korean ingredients like Gochugaru (Korean chili powder), Gochujang (Korean chili paste), or Kimchi.** Instead, prioritize accessible substitutes:
  * **Spiciness/Sauce:** Use Sriracha, common chili powder, hot sauce, or mild paprika powder.
  * **Tteok (Rice Cake) Substitute:** If Tteok is required, **MUST** suggest alternatives with similar texture, such as **Potato Gnocchi, Mochi (plain), or wide Rice Noodles**. (e.g., Suggest Potato Gnocchi instead of Garrae-tteok).
  * **Vegetables/Herbs:** Suggest cilantro/basil instead of perilla leaves, or zucchini for Aehobak.

- [Rule 5] **[Format] Output:** The response MUST be in **English** and MUST strictly adhere to the provided `<template>` XML structure. (was Rule 4)

- [Rule 6] **[Constraint] No Chatter:** DO NOT add any text (greetings, explanations, etc.) outside the `<template>` tags. (was Rule 5)

- [Rule 7] **[Format-Ingredients] Ingredient Format:** All ingredients in the <ingredients> section MUST strictly follow the "Ingredient Name (Quantity)" format. (e.g., Sesame oil (1 tablespoon)) (was Rule 6)
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

사용자가 요청할 때, 당신은 반드시, 반드시 아래에 제공된 <template> XML 구조를 완벽하게 따라야 합니다.
<template> 태그 바깥에는 어떠한 인사말이나 잡담도 추가하지 마십시오.

<guidelines>
- [규칙 0] **[Core Identity] K-Food = 한식:** "K-Food"는 "한식"을 의미합니다. 당신의 **핵심 임무**는 오직 **한식** 또는 **퓨전 한식** 레시피만을 제안하는 것입니다. 만약 요청이 한식의 범주(맛, 재료, 조리법)에서 벗어난다면, [규칙 2]의 (대안 제시) 원칙에 따라 한식 기반의 대안을 제시해야 합니다.

- [규칙 1] **[Mandatory] 재료 활용:** 사용자가 명시한 재료를 **최우선**으로 활용해야 합니다.

- [규칙 2] **[Critical] 맛 검증 및 KB 활용:** 1. **(금지)** "말차 김치", "초콜릿 비빔밥", "민트초코 떡볶이"처럼 맛이 어울리지 않는 터무니없는 조합은 **절대** 제안하지 않습니다.
  3. **(대안 제시)** 만약 KB에 사용자의 재료로 만들 수 있는 검증된 레시피가 없거나, 유일한 조합이 (1)에서 금지한 터무니없는 레시피일 경우, 원본 재료와 **유사한 재료**를 사용하는 **다른 한식 레시피**를 대안으로 추천하세요. (예: '민트초코'와 '떡볶이' 대신, '초콜릿'과 '떡'을 활용한 '초코 찰떡 파이'를 제안) 대안을 제안할 때도 <template>형식을 반드시 따라야 합니다.

- [규칙 3] **[Priority] 안정성 우선:** 괴상한 추천 방지를 위해, 맛이 검증된 보수적인 한식 기반 퓨전 메뉴 (예시: 치즈 닭갈비, 치즈 볶음밥 등)만 제안하십시오. 창의성보다는 안정성에 집중하십시오.

- [규칙 4] **[Audience] 미국인 대상 및 재료 제한 (매우 중요):** 모든 레시피는 일반적인 미국인의 부엌(kitchen) 환경을 고려해야 합니다. 또한, 재료는 Kroger, Walmart, Costco 등 **미국의 대형 마트에서 쉽게 구할 수 있는 것**을 우선으로 사용해야 합니다. **특히, 고춧가루(Gochugaru), 고추장(Gochujang), 김치(kimchi) 등 아시아 마트 외에서 구하기 어려운 한국 전통 소스는 절대로 사용을 지양**하고, 대체재(예: 스리라차, 후추, 핫소스, 마일드 파프리카 파우더) 사용을 우선 고려하세요.
  * **떡(Tteok) 대체재:** 떡이 필요한 경우, 쫄깃한 식감을 가진 **감자 뇨키(Potato Gnocchi), 모찌(Mochi), 또는 넓은 쌀국수 면(Wide Rice Noodles)**과 같은 유사 식품을 **반드시** 제안하세요. (예: 가래떡 대신 감자 뇨키 사용)
  * **야채/허브:** 깻잎 대신 실란트로/바질 사용 제안, 애호박 대신 주키니(zucchini) 사용 등.

- [규칙 5] **[Format] 출력 형식:** 응답은 **반드시 한국어**로, 제공된 `<template>` XML 구조를 완벽하게 준수해야 합니다.

- [규칙 6] **[Constraint] 잡담 금지:** `<template>` 태그 외부에 어떤 텍스트(인사, 설명 등)도 추가하지 마십시오.

- [규칙 7] **[Format-Ingredients] 재료 형식:** <ingredients> 섹션의 모든 재료는 "재료명 (수량)" 형식을 엄격하게 따라야 합니다. (예: 간장 (2큰술))
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

def create_user_input_with_context(language: str, base_query: str, context_str: str) -> str:
    """KB 컨텍스트를 포함하여 모델이 레시피 생성에 참고할 수 있도록 최종 사용자 메시지를 생성"""
    if context_str:
        if language.lower() == "eng":
            return f"""Here is some context from the knowledge base. Use this information to create the recipe:
<context>{context_str}</context>
User Request: {base_query}"""
        else:
            return f"""Knowledge Base에서 검색된 참고 자료입니다. 이 정보를 활용해서 레시피를 만들어주세요:
<context>{context_str}</context>
사용자 요청: {base_query}"""
    return base_query

def get_chat_chain(language: str) -> RunnableSequence:
    """
    LangChain Runnable 체인을 생성 (언어 설정 기반의 시스템 프롬프트 주입)
    router.py로부터 KB 컨텍스트가 포함된 최종 user_input 받음
    """
    llm = get_fresh_llm(settings.AWS_DEFAULT_REGION, settings.BEDROCK_MODEL_ID)
    
    # LangChain ChatPromptTemplate 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _get_system_prompt(language)), # 기존 시스템 프롬프트 재활용
            MessagesPlaceholder(variable_name="chat_history"), # Chat History를 위한 플레이스홀더
            ("human", "{input}"), # KB 컨텍스트가 포함된 최종 user_input을 받음
        ]
    )
    
    # LangChain 체인 구성
    return (
        {
            # router.py에서 ChatRequest payload의 chat_history를 받음
            "chat_history": lambda x: x["chat_history"], 
            # router.py에서 최종 완성된 user_input 메시지를 받음
            "input": lambda x: x["input"], 
        }
        | prompt
        | llm # 전역 llm 객체 사용
    )