from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from typing import List, Dict, Any, Optional, Iterator
import boto3
import json
from app.schemas.recipe import ChatRequest, ChatResponse, HotRecipe, TopIngredient
from app.services import bedrock_service, db_service
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableSequence
from app.core.config import settings

# bedrock_service에서 전역 객체를 직접 참조
llm = bedrock_service.llm
retriever = bedrock_service.retriever
MODEL_ID = bedrock_service.MODEL_ID # bedrock_service에서 로드된 전역 MODEL_ID 사용

router = APIRouter()

def lang_chain_stream_caller(
    chain: RunnableSequence, # LangChain 체인 객체
    chat_history: List[Dict[str, str]],
    user_input: str
) -> Iterator[str]:
    """
    LangChain의 동기 stream() 메서드를 실행하고 텍스트만 yield하는 Helper 함수.
    run_in_threadpool에 의해 스레드풀에서 실행됩니다.
    """
    
    # LangChain Chat History 형식으로 변환 (HumanMessage, AIMessage)
    lc_chat_history = []
    for msg in chat_history:
        if msg['role'] == 'user':
            lc_chat_history.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            lc_chat_history.append(AIMessage(content=msg['content']))

    # LangChain stream() 실행
    try:
        for chunk in chain.stream(
            {
                "chat_history": lc_chat_history,
                "input": user_input, 
            }
        ):
            # LangChain chunk 객체에서 content만 추출하여 yield
            if chunk.content:
                yield chunk.content
    except Exception as e:
        print(f"[LangChainStream] 오류 발생: {e}")
        yield f"<error>스트리밍 중 LangChain 오류 발생: {e}</error>"


@router.post("/chat/stream", tags=["Chat"])
async def handle_chat_stream(
    payload: ChatRequest,
):
    """
    (기능 1) LangChain 기반 Bedrock 챗봇 스트리밍 API (Chat History 및 KB 통합)
    """
    if not bedrock_service.llm: # 🔴 LLM 객체 존재 여부 확인 (bedrock_service에 llm이 있다고 가정)
        async def error_stream():
            yield "<error>LangChain LLM/Bedrock 서비스 초기화 실패. 설정을 확인하세요.</error>"
        return StreamingResponse(error_stream(), media_type="text/plain")

    language = payload.language
    ingredients = payload.ingredients
    chat_history = payload.chat_history or []
    is_first_message = not chat_history

    context_str = ""
    user_message = ingredients[0] if ingredients else "레시피 추천" # 꼬리 질문/첫 질문 텍스트

    # --- 1. KB 검색 (첫 질문일 때만) ---
    if is_first_message and ingredients and retriever:
        ingredient_list = ", ".join(ingredients)
        base_query = f"K-Food recipe using: {ingredient_list}" if language.lower() == "eng" else f"재료: {ingredient_list} K-Food 레시피"
        
        try:
            print(f"🔍 [KB] 비동기 검색 실행: {base_query}")
            # 동기 함수(retriever.invoke)를 비동기(FastAPI)에서 안전하게 실행
            retrieved_docs = await run_in_threadpool(retriever.invoke, base_query)
            context_str = bedrock_service.format_docs(retrieved_docs)
        except Exception as e:
            print(f"⚠️ [KB] Retriever failed: {e}")
            context_str = "Knowledge Base retrieval failed." if language.lower() == "eng" else "Knowledge Base 검색에 실패했습니다."
            
        # 첫 질문의 사용자 메시지를 KB 컨텍스트와 함께 재구성
        user_input_with_context = bedrock_service.create_user_input_with_context(
            language, base_query, context_str
        )
        # 🔴 LangChain 체인에 전달할 최종 입력 메시지
        final_input_message = user_input_with_context
    else:
        # 꼬리 질문일 경우, payload.ingredients[0] (실제 질문)을 사용
        final_input_message = user_message

    # --- 2. LangChain 체인 호출 및 스트리밍 ---
    try:
        # LangChain 체인 가져오기 (bedrock_service에 정의되어 있다고 가정)
        chain = bedrock_service.get_chat_chain(language, final_input_message) # 🔴 체인 생성 함수 호출

        # 🔴 run_in_threadpool로 LangChain 동기 스트림을 실행
        stream_iterator = await run_in_threadpool(
            lang_chain_stream_caller,
            chain,
            chat_history, # Chat History 전달
            final_input_message # 최종 사용자 입력 메시지 전달
        )

        # 🔴 StreamingResponse에 동기 제너레이터를 전달
        return StreamingResponse(stream_iterator, media_type="text/plain")

    except Exception as e:
        error_message = f"[LangChain] 치명적인 API 호출 오류: {e}"
        async def error_stream():
            yield f"<error>{error_message}</error>" 
        return StreamingResponse(error_stream(), media_type="text/plain")       

@router.get("/hot-recipes", response_model=List[Dict[str, Any]], tags=["Hot Recipes"])
async def get_hot_recipes():
    """
    (기능 2) Hot K-Food 추천 API
    DB(SQLite)에 저장된 Top 15 메뉴 중 랜덤 4개를 조회
    """
    recipes = await db_service.get_hot_recipes_from_db(limit=15)
    return recipes

@router.get("/hot-recipes/all", response_model=List[Dict[str, Any]], tags=["Hot Recipes"])
async def get_hot_recipes_all(): # 함수명 충돌 방지
    """
    secret API: DB에 저장된 모든 메뉴를 조회
    """
    recipes = await db_service.get_all_recipes_from_db()
    return recipes

@router.get("/hot-recipes/detail", response_model=Dict[str, Any], tags=["Hot Recipes"])
async def get_hot_recipes_detail(ranking: int):
    """
    (기능 2) Hot K-Food 추천 API
    DB(SQLite)에 저장된 메뉴의 디테일을 ranking을 통해 조회
    """
    recipe = await db_service.get_hot_recipes_detail_from_db(ranking=ranking)
    return recipe

@router.get("/top-ingredients", response_model=List[TopIngredient], tags=["Top Ingredients"])
async def get_top_ingredients():
    """
    (기능 3) Grocery 추천 API
    DB(SQLite)에 저장된 Top 10 재료를 조회
    """
    ingredients = await db_service.get_top_ingredients_from_db(limit=10)
    return ingredients