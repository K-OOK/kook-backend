from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from typing import List, Dict, Any
from app.schemas.recipe import ChatRequest, ChatResponse, HotRecipe, TopIngredient
from app.services import bedrock_service, db_service
from langchain_aws import AmazonKnowledgeBasesRetriever
from typing import List, Dict, Any, Optional
import boto3
import json
import os
import sys
from app.core.config import settings

BEDROCK_MODEL_ID = settings.BEDROCK_MODEL_ID

router = APIRouter()
@router.post("/chat/stream", tags=["Chat"])
async def handle_chat_stream(
    payload: ChatRequest,
):
    """
    (기능 1) Bedrock 챗봇 스트리밍 API
    유저의 재료와 언어 설정을 받아 Bedrock으로 레시피를 생성
    language: "kor" (한국어) 또는 "eng" (영어)
    """
    bedrock_runtime = bedrock_service.bedrock_runtime
    retriever = bedrock_service.retriever
    
    if not bedrock_runtime:
        async def error_stream():
            yield "<error>Bedrock client initialization failed. Check Cloud9 IAM permissions.</error>"
        return StreamingResponse(error_stream(), media_type="text/plain")

    language = payload.language
    ingredients = payload.ingredients
    chat_history = payload.chat_history or []
    is_first_message = not chat_history # 꼬리 질문 유지를 위한 핵심

    context_str = ""
    base_query = "" # KB 검색 쿼리

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
    
    # --- 2. Bedrock Payload 생성 ---
    # KB 컨텍스트와 Chat History를 포함한 최종 payload 생성
    try:
        # 1. Helper 함수 호출
        payload_data = bedrock_service.create_bedrock_payload(
            language=language,
            ingredients=ingredients,
            chat_history=chat_history,
            context_str=context_str,
        )
        
        # 🔴 2. Payload 분리: Boto3 호출에 필요한 두 인자를 추출
        payload_body = payload_data['bedrock_request_body']
        model_id = payload_data['model_id']

    except Exception as e:
        error_message = f"Payload 생성 오류: {e}" 
        
        async def error_stream():
            yield f"<error>{error_message}</error>" 
            
        return StreamingResponse(error_stream(), media_type="text/plain")
        
    # --- 3. Bedrock 스트리밍 API 호출 ---
    try:
        response_stream = await run_in_threadpool(
            bedrock_runtime.invoke_model_with_response_stream,
            modelId=model_id, # 🔴 분리된 model_id 사용
            body=json.dumps(payload_body) # 🔴 분리된 request_body 사용
        )

        # --- 4. 비동기 제너레이터 (스트리밍 응답 반환) ---
        async def stream_generator():
            try:
                if response_stream:
                    for event in response_stream.get("body"):
                        chunk = json.loads(event.get("chunk", {}).get("bytes", "{}"))
                        
                        if chunk.get('type') == 'content_block_delta':
                            text_delta = chunk.get('delta', {}).get('text', '')
                            yield text_delta
                        
                        elif chunk.get('type') == 'message_stop':
                            break
            except Exception as e:
                yield f"<error>스트리밍 중 오류 발생: {e}</error>"

        return StreamingResponse(stream_generator(), media_type="text/plain")

    except Exception as e:
        error_message = f"[Bedrock_Service] Bedrock API 호출 오류: {e}"
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
async def get_hot_recipes():
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