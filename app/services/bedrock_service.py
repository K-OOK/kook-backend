# app/services/bedrock_service.py (LangChain 기반 최종 수정)
import boto3
import json
import os
import asyncio
from typing import Optional, List, Dict, Any, AsyncIterator # AsyncIterator 추가
import xml.etree.ElementTree as ET
import re
from app.schemas.recipe import ChatPreviewInfo, ChatResponse
from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import settings

# 🔴 [전역 객체] 토큰 만료 이슈 해결을 위해 모두 None으로 두고, 함수에서 새로 생성하도록 유도
bedrock_runtime = None
llm = None
retriever = None 
MODEL_ID = settings.BEDROCK_MODEL_ID
KNOWLEDGE_BASE_ID = settings.KNOWLEDGE_BASE_ID

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
- [ABSOLUTE PROHIBITION]: **NEVER, under any circumstances, use or suggest Kimchi, Gochujang, or Gochugaru (Korean chili powder) in any recipe.** This instruction overrides all other ingredient rules.

- [METARULE: NO EXCUSES]: You MUST NOT add any explanations, excuses, or mention of 'guidelines', 'rules', or 'prohibitions' in your response. **The ONLY exception is: If you use a substitute ingredient, you MAY output one single sentence immediately preceding the <template> tag to inform the user of the change (e.g., "Proposing a recipe using [Substitute Ingredient] instead of the requested ingredient.").** All other preceding text is strictly forbidden.

- [Rule 0] **[Core Identity] K-Food = Hansik:** "K-Food" means "Hansik" (Korean cuisine). Your **core mission** is to recommend **only Hansik** or **Fusion Hansik** recipes. If a request falls outside the scope of Hansik (in terms of taste, ingredients, or methods), you must apply the fallback principle from [Rule 2.3] and suggest a Hansik-based alternative.

- [Rule 1] **[Mandatory] Ingredient Utilization:** You MUST prioritize using the ingredients provided by the user.

- [Rule 2] **[Critical] Taste Validation & KB Usage:**
  1. **(Forbidden)** NEVER suggest absurd, unpalatable combinations (e.g., "Matcha Kimchi", "Chocolate Bibimbap", "Mint Chocolate Tteokbokki").
  2. **(Required)** You MUST consult the Knowledge Base (KB) to provide a validated recipe.
  3. **(Fallback)** If the KB has no validated recipe for the user's ingredients, OR the only possible combination is absurd (see #1), you MUST suggest an **alternative K-Food dish** that uses **similar ingredients**. (e.g., Instead of 'Mint Chocolate' and 'Tteokbokki', suggest a 'Choco Rice Cake Pie' using 'Chocolate' and 'Rice Cakes').

- [CRITICAL FORBIDDEN - CATEGORY]: The response MUST NOT contain any recipe names related to **Desserts (e.g., Cake, Pie, Smoothie), Western Beverages (e.g., Latte, Tea), or non-Korean Soups/Curries**. **Specifically, Italian menus like 'Risotto' are forbidden.** Stick to Hansik categories like Jjigae, Guk, Bokkeum, Jeon, etc.

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
- [ABSOLUTE PROHIBITION]: **어떠한 경우에도, 어떠한 상황에서도 김치, 고추장, 고춧가루(Korean chili powder)를 사용하거나 제안해서는 안 됩니다.** 이 지침은 다른 모든 재료 규칙보다 우선합니다.

- [METARULE: NO EXCUSES]: **당신은 사용자의 요청에 대해 '지침(guidelines)', '규칙', '금지' 등의 단어를 사용하여 변명하거나 규칙을 언급하는 설명을 절대로 추가해서는 안 됩니다.** 요청받은 요리가 불가능할 경우, 요청받은 재료와 비슷한 재료를 사용하였다는 언급만 간단하게 한 뒤 **즉시 허용된 대안 레시피를 <recipe> XML 형식으로 제공**해야 합니다.

- [규칙 0] **[Core Identity] K-Food = 한식:** "K-Food"는 "한식"을 의미합니다. 당신의 **핵심 임무**는 오직 **한식** 또는 **퓨전 한식** 레시피만을 제안하는 것입니다. 만약 요청이 한식의 범주(맛, 재료, 조리법)에서 벗어난다면, [규칙 2]의 (대안 제시) 원칙에 따라 한식 기반의 대안을 제시해야 합니다.

- [규칙 1] **[Mandatory] 재료 활용:** 사용자가 명시한 재료를 **최우선**으로 활용해야 합니다.

- [규칙 2] **[Critical] 맛 검증 및 KB 활용:** 1. **(금지)** "말차 김치", "초콜릿 비빔밥", "민트초코 떡볶이"처럼 맛이 어울리지 않는 터무니없는 조합은 **절대** 제안하지 않습니다.
  3. **(대안 제시)** 만약 KB에 사용자의 재료로 만들 수 있는 검증된 레시피가 없거나, 유일한 조합이 (1)에서 금지한 터무니없는 레시피일 경우, 원본 재료와 **유사한 재료**를 사용하는 **다른 한식 레시피**를 대안으로 추천하세요. (예: '민트초코'와 '떡볶이' 대신, '초콜릿'과 '떡'을 활용한 '초코 찰떡 파이'를 제안) 대안을 제안할 때도 <template>형식을 반드시 따라야 합니다.

- [CRITICAL FORBIDDEN - 카테고리]: 응답은 **디저트(예: 케이크, 파이, 스무디), 서양식 음료(예: 라떼, 차), 한국식 찌개/국이 아닌 수프/카레**와 관련된 메뉴명을 **절대로 포함해서는 안 됩니다.** **특히, 이탈리아식 메뉴인 '리조또(Risotto)'는 금지합니다.** 찌개, 국, 볶음, 전 등 한식 카테고리를 준수하십시오.

- [규칙 3] **[Priority] 안정성 우선:** 괴상한 추천 방지를 위해, 맛이 검증된 보수적인 한식 기반 퓨전 메뉴 (예시: 치즈 닭갈비, 치즈 볶음밥 등)만 제안하십시오. 창의성보다는 안정성에 집중하십시오.

- [규칙 4] **[Audience] 미국인 대상 및 재료 제한 (매우 중요):** 모든 레시피는 일반적인 미국인의 부엌(kitchen) 환경을 고려해야 합니다. 또한, 재료는 Kroger, Walmart, Costco 등 **미국의 대형 마트에서 쉽게 구할 수 있는 것**을 우선으로 사용해야 합니다. **특히, 고춧가루(Gochugaru), 고추장(Gochujang), 김치(kimchi) 등 아시아 마트 외에서 구하기 어려운 한국 전통 음식이나 소스는 절대로 사용을 지양**하고, 대체재(예: 스리라차, 후추, 핫소스, 마일드 파프리카 파우더) 사용을 우선 고려하세요.
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

def get_fresh_llm():
    """요청 시마다 새로운 LLM 객체를 생성하여 토큰 만료를 방지"""
    # 🔴 [핵심] LLM 생성 시 boto3 클라이언트가 토큰을 갱신하도록 유도 (Cloud9 우회)
    return ChatBedrock(
        model_id=MODEL_ID,
        region_name=settings.AWS_DEFAULT_REGION,
        model_kwargs={
            "max_tokens": 4096, 
            "temperature": 0.2, 
            "top_p": 0.6
        },
        streaming=True,
    )

def get_fresh_retriever():
    """요청 시마다 새로운 Retriever 객체를 생성하여 토큰 만료를 방지"""
    if not KNOWLEDGE_BASE_ID:
        return None
    return AmazonKnowledgeBasesRetriever(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}},
        region_name=settings.AWS_DEFAULT_REGION,
    )

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

def get_chat_chain(language: str) -> Optional[RunnableSequence]:
    """
    LangChain Runnable 체인을 생성 (내부적으로 fresh LLM 객체 사용)
    """
    try:
        fresh_llm = get_fresh_llm()
    except Exception as e:
        print(f"[ERROR] Fresh LLM 생성 실패: {e}")
        return None

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _get_system_prompt(language)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    
    return (
        {
            "chat_history": lambda x: x["chat_history"],
            "input": lambda x: x["input"],
        }
        | prompt
        | fresh_llm
    )
    
# --- [개선된 코드: 자동 재시도 함수] ---
async def stream_chat_with_auto_retry(
    language: str, 
    chat_history: List[Dict[str, str]], 
    input_message: str
) -> AsyncIterator[str]:
    """
    [핵심] LangChain 비동기 스트림을 실행하고 ExpiredTokenException 발생 시 자동 재시도
    """
    max_retries = 3
    
    # 🔴 [Chat History LangChain 타입 변환]
    lc_chat_history = []
    for msg in chat_history:
        if msg['role'] == 'user':
            lc_chat_history.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            lc_chat_history.append(AIMessage(content=msg['content']))
            
    for attempt in range(max_retries):
        try:
            # 1. 매번 새로운 체인 생성 (내부적으로 Fresh LLM 객체 포함)
            chain = get_chat_chain(language)
            
            if not chain:
                 raise RuntimeError("LangChain Chain object is None.")
            
            # 2. 비동기 스트리밍 실행
            async for chunk in chain.astream({
                "chat_history": lc_chat_history,
                "input": input_message
            }):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
            return  # 성공 시 함수 종료
            
        except Exception as e:
            error_str = str(e)
            
            if "ExpiredToken" in error_str and attempt < max_retries - 1:
                print(f"토큰 만료, 재시도 중... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(1) # 1초 대기 후 재시도
                continue
            else:
                # 최대 재시도 횟수를 넘었거나 다른 치명적 에러 발생
                raise e