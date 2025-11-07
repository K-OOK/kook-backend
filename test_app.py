import streamlit as st
import boto3
import json
import os
import re
import sys
import asyncio
import xml.etree.ElementTree as ET
from typing import Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_aws import AmazonKnowledgeBasesRetriever

# --- [0] Streamlit 테스트를 위한 준비 ---

# 커맨드라인 인자 파싱 (streamlit run test_app.py -- arg1 arg2 arg3 arg4 형태)
# Streamlit은 sys.argv를 다르게 처리할 수 있으므로 여러 방법 시도

# 디버깅: sys.argv 전체 출력
print(f"[디버깅] sys.argv 전체: {sys.argv}")
print(f"[디버깅] sys.argv 길이: {len(sys.argv)}")
print(f"[디버깅] __file__: {__file__}")

APP_ARGS = []

# 방법 1: '--' 구분자 사용
if '--' in sys.argv:
    idx = sys.argv.index('--')
    APP_ARGS = sys.argv[idx + 1:]
    print(f"[Streamlit] 방법 1 - 커맨드라인 인자 (-- 이후): {APP_ARGS}")

# 방법 2: 스크립트 파일명 이후의 인자 찾기
if not APP_ARGS:
    script_name = os.path.basename(__file__)
    script_path = __file__
    
    # sys.argv에서 스크립트 경로나 이름 찾기
    for i, arg in enumerate(sys.argv):
        if script_name in arg or script_path in arg:
            if i + 1 < len(sys.argv):
                # '--' 이후의 인자만 가져오기
                remaining = sys.argv[i + 1:]
                if '--' in remaining:
                    idx = remaining.index('--')
                    APP_ARGS = remaining[idx + 1:]
                else:
                    APP_ARGS = remaining
                print(f"[Streamlit] 방법 2 - 커맨드라인 인자 (스크립트 이후): {APP_ARGS}")
                break

# 방법 3: sys.argv[1:]에서 '--' 이후 찾기
if not APP_ARGS and len(sys.argv) > 1:
    remaining = sys.argv[1:]
    if '--' in remaining:
        idx = remaining.index('--')
        APP_ARGS = remaining[idx + 1:]
        print(f"[Streamlit] 방법 3 - 커맨드라인 인자: {APP_ARGS}")

if not APP_ARGS:
    print(f"[Streamlit] ⚠️ 커맨드라인 인자를 찾을 수 없습니다.")

# 커맨드라인 인자 파싱 및 검증
# arg1: language (kor/eng) - required
# arg2: ingredient1 - required
# arg3: ingredient2 - optional
# arg4: ingredient3 - optional
PARSED_ARGS = {
    "language": None,
    "ingredients": []
}

if len(APP_ARGS) >= 2:
    # arg1: language 검증
    arg1 = APP_ARGS[0].lower()
    if arg1 in ["kor", "eng"]:
        PARSED_ARGS["language"] = arg1
    else:
        print(f"⚠️ [인자 오류] arg1은 'kor' 또는 'eng'이어야 합니다. 현재 값: {arg1}")
    
    # arg2: ingredient1 (required)
    PARSED_ARGS["ingredients"].append(APP_ARGS[1])
    
    # arg3: ingredient2 (optional)
    if len(APP_ARGS) >= 3:
        PARSED_ARGS["ingredients"].append(APP_ARGS[2])
    
    # arg4: ingredient3 (optional)
    if len(APP_ARGS) >= 4:
        PARSED_ARGS["ingredients"].append(APP_ARGS[3])
    
    print(f"[Streamlit] 파싱된 인자: language={PARSED_ARGS['language']}, ingredients={PARSED_ARGS['ingredients']}")
elif len(APP_ARGS) > 0:
    print(f"⚠️ [인자 오류] 최소 2개의 인자가 필요합니다 (language, ingredient1). 현재 인자 개수: {len(APP_ARGS)}")

print(f"[Streamlit] PARSED_ARGS: {PARSED_ARGS}")

# 1. .env 파일 로드
load_dotenv()

# 2. FastAPI의 'settings' 객체 대신 os.environ에서 직접 값 로드
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID") # .env에서 로드

# 3. FastAPI의 'schemas' 모듈 대신 dataclass로 모조품(shim) 생성
@dataclass
class ChatPreviewInfo:
    """app.schemas.recipe.ChatPreviewInfo의 모조품"""
    total_time: str
    ingredients: List[str]

@dataclass
class ChatResponse:
    """app.schemas.recipe.ChatResponse의 모조품"""
    full_recipe: str
    preview: Optional[ChatPreviewInfo]

# --- [1] 네가 제공한 코드 (settings 부분만 위 변수로 수정) ---

# 설정 파일에서 AWS 정보 로드
try:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_DEFAULT_REGION, # settings.AWS_DEFAULT_REGION -> AWS_DEFAULT_REGION
    )   
    MODEL_ID = BEDROCK_MODEL_ID # settings.BEDROCK_MODEL_ID -> BEDROCK_MODEL_ID
    
    if KNOWLEDGE_BASE_ID: # settings.KNOWLEDGE_BASE_ID -> KNOWLEDGE_BASE_ID
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}},
            region_name=AWS_DEFAULT_REGION, # settings.AWS_DEFAULT_REGION -> AWS_DEFAULT_REGION
        )
        print(f"[Streamlit] Retriever for KB ID: {KNOWLEDGE_BASE_ID} initialized.")
    else:
        retriever = None
        print("[Streamlit] KNOWLEDGE_BASE_ID not found. Retriever is None.")

except Exception as e:
    print(f"[Bedrock_Service] Boto3 클라이언트 또는 Retriever 초기화 실패: {e}")
    st.error(f"Boto3/Retriever 초기화 실패: {e}") # Streamlit UI에도 에러 표시
    bedrock_runtime = None
    retriever = None
    MODEL_ID = None
    KNOWLEDGE_BASE_ID = None


def format_docs(docs):
    """검색된 문서를 문자열로 변환 (제공한 코드와 동일)"""
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

    result = (
        "\n\n---\n\n".join(formatted)
        if formatted
        else ""
    )
    print(f"✅ [KB] {len(formatted)}개 문서 포맷 완료 (총 {len(result)}자)")
    return result


def _get_system_prompt(language: str) -> str:
    """language에 따라 한국어 또는 영어 시스템 프롬프트를 반환 (제공한 코드와 동일)"""
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
<title>[ Write the dish title here ] (for 1 serving)</title>
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
- [규칙 1] **[Mandatory] 재료 활용:** 사용자가 명시한 재료를 **최우선**으로 활용해야 합니다.

- [규칙 2] **[Critical] 맛 검증 및 KB 활용:** 1. **(금지)** "말차 김치", "초콜릿 비빔밥", "민트초코 떡볶이"처럼 맛이 어울리지 않는 터무니없는 조합은 **절대** 제안하지 않습니다.
  2. **(필수)** 레시피 제안 시 **반드시** Knowledge Base(KB)의 정보를 참고하여 검증된 레시피를 제공해야 합니다.
  3. **(대안 제시)** 만약 KB에 사용자의 재료로 만들 수 있는 검증된 레시피가 없거나, 유일한 조합이 (1)에서 금지한 터무니없는 레시피일 경우, 원본 재료와 **유사한 재료**를 사용하는 **다른 한식 레시피**를 대안으로 추천하세요. (예: '민트초코'와 '떡볶이' 대신, '초콜릿'과 '떡'을 활용한 '초코 찰떡 파이'를 제안)

- [규칙 3] **[Priority] 검증된 퓨전:** '고추장 버터 불고기', '김치 치즈 파스타', '콘치즈 닭갈비'처럼 (맛이 검증된) 창의적인 퓨전 요리를 **우선적으로** 제안하세요.

- [규칙 4] **[Format] 출력 형식:** 응답은 **반드시 한국어**로, 제공된 `<template>` XML 구조를 완벽하게 준수해야 합니다.

- [규칙 5] **[Constraint] 잡담 금지:** `<template>` 태그 외부에 어떤 텍스트(인사, 설명 등)도 추가하지 마십시오.
</guidelines>

<template>
<recipe>
<title>[ 여기에 요리 제목을 적어주세요 ] (1_serving 기준)</title>
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
    """제공한 코드와 동일 (ChatPreviewInfo 스키마만 dataclass로 대체)"""
    try:
        if '<recipe>' in xml_string:
            xml_string = "<recipe>" + xml_string.split('<recipe>', 1)[1]
        if '</recipe>' in xml_string:
            xml_string = xml_string.split('</recipe>', 1)[0] + "</recipe>"
            
        root = ET.fromstring(xml_string)
        is_english = language.lower() == "eng"
        
        ingredients_list = []
        if is_english:
            ingredients_section = root.find(".//section[title='1. Ingredients 🥣']")
        else:
            ingredients_section = root.find(".//section[title='1. 재료 🥣']")
        
        if ingredients_section is not None:
            ingredients_tag = ingredients_section.find('ingredients')
            if ingredients_tag is not None and ingredients_tag.text:
                ingredients_list = [
                    line.strip() for line in ingredients_tag.text.strip().split('\n') 
                    if line.strip()
                ]

        total_time = "정보 없음" if not is_english else "Information not available"
        if is_english:
            steps_section_title = root.find(".//section/title[starts-with(., '2. Cooking Method 🍳')]")
            if steps_section_title is not None and steps_section_title.text:
                match = re.search(r'\((Total estimated time:.*?)\)', steps_section_title.text)
                if match:
                    total_time = match.group(1)
        else:
            steps_section_title = root.find(".//section/title[starts-with(., '2. 조리 방법 🍳')]")
            if steps_section_title is not None and steps_section_title.text:
                match = re.search(r'\((총 예상 시간:.*?)\)', steps_section_title.text)
                if match:
                    total_time = match.group(1)

        return ChatPreviewInfo(
            total_time=total_time,
            ingredients=ingredients_list
        )
        
    except Exception as e:
        print(f"[XML 파싱 오류] {e}")
        return None


def generate_chat_response(user_message: str, language: str, chat_history: List[dict] = None, is_first_message: bool = False):
    """
    챗봇 형태로 대화를 생성하는 함수
    chat_history: 이전 대화 기록 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    is_first_message: 첫 번째 메시지인지 여부 (KB 검색 여부 결정)
    """
    if not bedrock_runtime:
        error_msg = "Bedrock service is not initialized."
        if language.lower() != "eng":
            error_msg = "Bedrock service가 초기화되지 않았습니다."
        return {"role": "assistant", "content": f"<error>{error_msg}</error>"}, "N/A"

    system_prompt = _get_system_prompt(language)
    is_english = language.lower() == "eng"
    
    # 첫 번째 메시지이고 재료가 포함된 경우에만 KB 검색
    context_str = ""
    if is_first_message and retriever:
        try:
            # 재료 추출 시도
            base_query = user_message if is_english else f"K-Food recipe: {user_message}"
            print(f"🔍 [KB] Retrieving for query: {base_query}")
            retrieved_docs = retriever.invoke(base_query)
            context_str = format_docs(retrieved_docs)
        except Exception as e:
            print(f"⚠️ [KB] Retriever failed: {e}")
            context_str = "Knowledge Base retrieval failed." if is_english else "Knowledge Base 검색에 실패했습니다."
    
    # 메시지 구성
    messages = []
    if chat_history:
        messages.extend(chat_history)
    
    # 현재 사용자 메시지에 컨텍스트 추가 (첫 번째 메시지이고 컨텍스트가 있는 경우만)
    if is_first_message and context_str:
        if is_english:
            full_user_message = f"""Here is some context from the knowledge base. Use this information to create the recipe:
<context>
{context_str}
</context>

User Request: {user_message}
"""
        else:
            full_user_message = f"""Knowledge Base에서 검색된 참고 자료입니다. 이 정보를 활용해서 레시피를 만들어주세요:
<context>
{context_str}
</context>

사용자 요청: {user_message}
"""
    else:
        full_user_message = user_message
    
    messages.append({"role": "user", "content": full_user_message})

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages
        })

        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=body
        )

        response_body = json.loads(response.get('body').read())
        
        content_list = response_body.get('content', [])
        if content_list and isinstance(content_list, list) and 'text' in content_list[0]:
            assistant_message = content_list[0].get('text')
        else:
            assistant_message = f"<error>Unexpected model response format: {response_body}</error>"

        return {"role": "assistant", "content": assistant_message}, context_str

    except Exception as e:
        print(f"[Bedrock_Service] Bedrock API 호출 오류: {e}")
        error_msg = f"An error occurred: {e}"
        if language.lower() != "eng":
            error_msg = f"레시피 생성 중 오류: {e}"
        
        return {"role": "assistant", "content": f"<error>{error_msg}</error>"}, context_str

# --- [2] Streamlit UI 부분 ---

st.set_page_config(layout="wide", page_title="셰프 김 챗봇")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "language" not in st.session_state:
    # 커맨드라인 인자에서 language가 있으면 사용, 없으면 기본값 "eng"
    st.session_state.language = PARSED_ARGS.get("language") or "eng"
if "kb_context" not in st.session_state:
    st.session_state.kb_context = ""
if "initial_ingredients" not in st.session_state:
    # 커맨드라인 인자에서 재료가 있으면 저장
    st.session_state.initial_ingredients = PARSED_ARGS.get("ingredients", [])

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 커맨드라인 인자 표시
    if APP_ARGS:
        st.subheader("커맨드라인 인자")
        st.code(" ".join(APP_ARGS), language="bash")
        st.caption("사용법: `streamlit run test_app.py -- language ingredient1 [ingredient2] [ingredient3]`")
        if PARSED_ARGS["language"]:
            st.success(f"✅ Language: `{PARSED_ARGS['language']}`")
        if PARSED_ARGS["ingredients"]:
            st.success(f"✅ Ingredients: `{', '.join(PARSED_ARGS['ingredients'])}`")
        st.divider()
    
    # .env 로드 상태 표시
    st.subheader("환경 변수")
    st.info(f"**Region:** `{AWS_DEFAULT_REGION}`")
    st.info(f"**Model ID:** `{BEDROCK_MODEL_ID}`")
    if KNOWLEDGE_BASE_ID:
        st.success(f"**KB ID:** `{KNOWLEDGE_BASE_ID}` ✅")
    else:
        st.warning("**KB ID:** `None` (KB 검색 비활성화)")
    
    st.divider()
    
    # 언어 설정 (커맨드라인 인자가 있으면 해당 값으로 초기화)
    default_lang_idx = 0 if st.session_state.language == "eng" else 1
    st.session_state.language = st.selectbox("언어 (Language)", ["eng", "kor"], index=default_lang_idx)
    
    st.divider()
    
    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.kb_context = ""
        st.rerun()


# 대화 기록 표시
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                # XML인 경우 코드 블록으로 표시
                content = message["content"]
                if content.startswith("<recipe>") or content.startswith("<error>"):
                    st.code(content, language="xml")
                else:
                    st.write(content)
                
                # 미리보기 정보 표시 (XML인 경우)
                if content.startswith("<recipe>"):
                    preview = _parse_recipe_xml_for_preview(content, st.session_state.language)
                    if preview:
                        with st.expander("📄 미리보기 정보"):
                            st.json(preview.__dict__)

# 커맨드라인 인자로 재료가 전달된 경우 자동으로 첫 메시지 생성
if "auto_sent" not in st.session_state:
    st.session_state.auto_sent = False

# 자동 메시지 생성 조건 확인
should_auto_send = (
    not st.session_state.auto_sent and 
    st.session_state.initial_ingredients and 
    len(st.session_state.initial_ingredients) > 0 and
    len(st.session_state.chat_history) == 0
)

if should_auto_send and bedrock_runtime:
    # 자동으로 첫 메시지 생성
    ingredients_str = ", ".join(st.session_state.initial_ingredients)
    if st.session_state.language == "eng":
        auto_message = f"Please create a K-Food recipe using these ingredients: {ingredients_str}"
    else:
        auto_message = f"내가 가진 재료: {ingredients_str}로 K-Food 레시피를 만들어주세요."
    
    print(f"[자동 메시지] 생성 중: {auto_message}")
    
    # 자동 메시지 처리
    is_first = True
    assistant_response, kb_context = generate_chat_response(
        auto_message,
        st.session_state.language,
        [],
        is_first_message=is_first
    )
    
    st.session_state.chat_history.append({"role": "user", "content": auto_message})
    if kb_context:
        st.session_state.kb_context = kb_context
    st.session_state.chat_history.append(assistant_response)
    st.session_state.auto_sent = True
    
    print(f"[자동 메시지] 완료. 대화 기록 개수: {len(st.session_state.chat_history)}")
    st.rerun()

# 사용자 입력
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    if not bedrock_runtime:
        st.error("Boto3 클라이언트가 초기화되지 않았습니다. AWS 설정을 확인하세요.")
    else:
        # 첫 번째 메시지인지 확인 (사용자 메시지 추가 전)
        is_first = len(st.session_state.chat_history) == 0
        
        # 응답 생성 (현재 대화 기록 사용)
        with st.spinner("생성 중..."):
            assistant_response, kb_context = generate_chat_response(
                user_input,
                st.session_state.language,
                st.session_state.chat_history,  # 현재까지의 대화 기록
                is_first_message=is_first
            )
            
            # 사용자 메시지를 대화 기록에 추가
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # KB 컨텍스트 저장 (첫 번째 메시지인 경우)
            if is_first and kb_context:
                st.session_state.kb_context = kb_context
            
            # 어시스턴트 응답을 대화 기록에 추가
            st.session_state.chat_history.append(assistant_response)
        
        # 페이지 새로고침하여 대화 기록 업데이트
        st.rerun()