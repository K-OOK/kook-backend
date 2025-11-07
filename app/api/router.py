from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from typing import List, Dict, Any, Optional
import boto3
import json
from app.schemas.recipe import ChatRequest, ChatResponse, HotRecipe, TopIngredient
from app.services import bedrock_service, db_service
from langchain_aws import AmazonKnowledgeBasesRetriever
from app.core.config import settings
import os
import sys

# bedrock_service에서 전역 객체를 직접 참조
bedrock_runtime = bedrock_service.bedrock_runtime
retriever = bedrock_service.retriever
MODEL_ID = bedrock_service.MODEL_ID # bedrock_service에서 로드된 전역 MODEL_ID 사용

router = APIRouter()

# 🔴 [새로운 Helper 함수 정의] Boto3 호출과 스트림 순회 전체를 담당하는 동기 함수
def sync_stream_caller(
    bedrock_runtime: boto3.client,
    model_id: str,
    payload_body: Dict[str, Any]
):
    """
    Boto3 호출부터 EventStream 순회까지 모든 동기 작업을 수행하는 헬퍼 함수.
    이 함수가 Threadpool에서 실행됩니다.
    """
    try:
        # 1. Boto3 호출 (동기)
        response_stream = bedrock_runtime.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps(payload_body, ensure_ascii=False)
        )
        
        # 2. EventStream을 동기 for 루프로 안전하게 순회합니다.
        for event in response_stream.get("body"):
            chunk = json.loads(event.get("chunk", {}).get("bytes", "{}"))
            
            if chunk.get('type') == 'content_block_delta':
                text_delta = chunk.get('delta', {}).get('text', '')
                yield text_delta # 동기 제너레이터로서 청크 반환
            
            elif chunk.get('type') == 'message_stop':
                break
                
    except Exception as e:
        print(f"[Bedrock_Service] 동기 스트림 호출 오류: {e}")
        yield f"<error>Bedrock API 호출 오류 (Threadpool): {e}</error>"


@router.post("/chat/stream", tags=["Chat"])
async def handle_chat_stream(
    payload: ChatRequest,
):
    """
    (기능 1) Bedrock 챗봇 스트리밍 API (Chat History 및 KB 포함)
    - 스트리밍 충돌 문제를 회피하기 위해 동기 헬퍼 함수를 사용합니다.
    """
    if not bedrock_runtime:
        async def error_stream():
            yield "<error>Bedrock client initialization failed. Check setup.</error>"
        return StreamingResponse(error_stream(), media_type="text/plain")

    language = payload.language
    ingredients = payload.ingredients
    chat_history = payload.chat_history or []
    is_first_message = not chat_history 

    context_str = ""
    
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
    try:
        # 1. Helper 함수 호출
        payload_data = bedrock_service.create_bedrock_payload(
            language=language,
            ingredients=ingredients,
            chat_history=chat_history,
            context_str=context_str,
        )
        # 2. Payload 분리 (ValidationException 회피)
        payload_body = payload_data['bedrock_request_body']
        model_id = payload_data['model_id']

    except Exception as e:
        error_message = f"Payload 생성 오류: {e}" 
        async def error_stream():
            yield f"<error>{error_message}</error>" 
        return StreamingResponse(error_stream(), media_type="text/plain")
        
    # --- 3. Bedrock 스트리밍 API 호출 및 응답 반환 ---
    try:
        # 🔴 StreamingResponse에 동기 제너레이터(sync_stream_caller)를 전달.
        #    FastAPI가 이를 Threadpool에서 실행하여 스트리밍을 처리합니다.
        stream_iterator = sync_stream_caller(
            bedrock_runtime,
            model_id,
            payload_body
        )

        return StreamingResponse(stream_iterator, media_type="text/plain")

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