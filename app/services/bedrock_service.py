import boto3
import json
from app.core.config import settings
from typing import Optional, List, Dict, Any
import xml.etree.ElementTree as ET
import re
from app.schemas.recipe import ChatPreviewInfo, ChatResponse
from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence # 체인 타입
from langchain_core.messages import HumanMessage, AIMessage # 메시지 타입

# 설정 파일에서 AWS 정보 로드
try:
    llm = ChatBedrock(
        model_id=settings.BEDROCK_MODEL_ID,
        region_name=settings.AWS_DEFAULT_REGION,
        model_kwargs={"max_tokens": 4096}, # 충분히 넉넉하게 설정
        streaming=True, # 스트리밍 활성화
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
    print(f"[Bedrock_Service] LangChain LLM 또는 Retriever 초기화 실패: {e}")
    bedrock_runtime = None
    llm = None # 🔴 LLM 객체 실패 시 None 할당
    retriever = None # 실패 시 retriever도 None
    MODEL_ID = None
    KNOWLEDGE_BASE_ID = None

def format_docs(docs):
    """검색된 문서를 문자열로 변환 (참고 코드에서 가져옴)"""
    if not docs:
        print("⚠️ [KB] 검색된 문서 없음")
        return "" # KB 검색 결과 없으면 빈 문자열 반환
    # ... (로직 생략, 기존과 동일) ...
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


# --- [_get_system_prompt] 함수는 그대로 유지 ---
def _get_system_prompt(language: str) -> str:
    # ... (로직 생략, 기존과 동일) ...
    if language.lower() == "eng":
        return """You are "Chef Kim", a professional chef who introduces K-Food to foreigners.
... (중략) ...
"""
    else:  # 한국어 (기본값)
        return """당신은 "셰프 김(Chef Kim)"이라는 이름을 가진, 외국인에게 **K-Food(한식)**를 알려주는 전문 요리사입니다.
... (중략) ...
"""

# --- [create_bedrock_payload 함수를 LangChain Helper로 대체] ---

def create_user_input_with_context(language: str, base_query: str, context_str: str) -> str:
    """KB 컨텍스트가 포함된 최종 사용자 메시지를 생성"""
    if context_str:
        if language.lower() == "eng":
            return f"""Here is some context. Use this to create the recipe:
<context>{context_str}</context>
User Request: {base_query}"""
        else:
            return f"""KB 참고 자료입니다:
<context>{context_str}</context>
사용자 요청: {base_query}"""
    return base_query


def get_chat_chain(language: str) -> RunnableSequence:
    """
    LangChain Runnable 체인을 생성 (LangChain 통합의 핵심)
    KB 검색 결과는 router.py에서 context_str로 이미 처리되었으므로,
    이 체인은 단순하게 프롬프트와 LLM을 결합합니다.
    """
    
    # LangChain ChatPromptTemplate 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _get_system_prompt(language)), # 기존 시스템 프롬프트 재활용
            MessagesPlaceholder(variable_name="chat_history"), # Chat History를 위한 플레이스홀더
            ("human", "{input}"),
        ]
    )
    
    # 🔴 LangChain 체인 구성
    return (
        {
            # chat_history와 input은 router에서 LangChain 형식에 맞게 payload로 전달
            "chat_history": lambda x: x["chat_history"],
            "input": lambda x: x["input"],
        }
        | prompt
        | llm # 🔴 전역 llm 객체 사용
    )