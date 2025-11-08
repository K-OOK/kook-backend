from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator # Iterator 추가
import boto3
import json
from app.schemas.recipe import ChatRequest, ChatResponse, HotRecipe, TopIngredient
from app.services import bedrock_service, db_service
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_core.messages import HumanMessage, AIMessage # LangChain 메시지 타입 추가
from langchain_core.runnables import RunnableSequence # LangChain 체인 타입 추가
from app.core.config import settings

router = APIRouter()

@router.post("/chat/stream", tags=["Chat"])
async def handle_chat_stream(
    payload: ChatRequest,
):
    """
    (기능 1) LangChain 기반 Bedrock 챗봇 스트리밍 API (Chat History 및 KB 통합)
    """
    # 🔴 LLM/Retriever 객체를 매번 새로 생성하므로, 여기서 초기화 검사는 불필요
    
    language = payload.language
    ingredients = payload.ingredients
    chat_history = payload.chat_history or []
    is_first_message = not chat_history

    context_str = ""
    
    # 🔴 [Retriever 생성] 요청 시마다 새로운 Retriever 객체 사용
    retriever = bedrock_service.get_fresh_retriever()
    
    base_query = ingredients[0] if ingredients and len(ingredients) > 0 else ("Recommend K-Food" if language.lower() == "eng" else "K-Food 추천") # 꼬리 질문/첫 질문 텍스트

    # --- 1. KB 검색 (첫 질문일 때만) ---
    if is_first_message and retriever:
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
        final_input_message = bedrock_service.create_user_input_with_context(
            language, base_query, context_str
        )
    else:
        # 꼬리 질문일 경우, payload.ingredients[0] (실제 질문)을 사용
        final_input_message = base_query


    # --- 2. Bedrock 스트리밍 호출 및 응답 반환 ---
    async def stream_generator_with_error_handling() -> AsyncIterator[str]:
        try:
            # 🔴 [핵심] 자동 재시도 기능이 있는 비동기 스트림 헬퍼 함수 호출
            async for chunk in bedrock_service.stream_chat_with_auto_retry(
                language, 
                chat_history, 
                final_input_message
            ):
                yield chunk
        except Exception as e:
            # 최종적으로 재시도가 실패했을 때의 에러 처리
            error_message = f"[LangChain] 치명적인 API 호출 오류 (재시도 실패): {e}"
            print(f"🚨 {error_message}")
            yield f"<error>{error_message}</error>"

    return StreamingResponse(
        stream_generator_with_error_handling(), 
        media_type="text/plain"
    )        

@router.get("/hot-recipes", response_model=List[Dict[str, Any]], tags=["Hot Recipes"])
async def get_hot_recipes():
    """
    (기능 2) Hot K-Food 추천 API
    DB(SQLite)에 저장된 Top 15 메뉴 중 랜덤 4개를 조회
    """
    recipes = await db_service.get_hot_recipes_from_db(limit=15)
    return recipes

@router.get("/hot-recipes/all", response_model=List[Dict[str, Any]], tags=["Hot Recipes"])
async def get_hot_recipes_all():
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