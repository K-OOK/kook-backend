import boto3
import os
import sys
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings # 👈 LangChain의 Bedrock 연동기능

# (주의) 이 스크립트는 app 모듈(config)을 사용하므로,
# 'python -m scripts.embed_pdfs'로 실행해야 함
try:
    from app.core.config import settings
except ModuleNotFoundError:
    print("---------------------------------------------------------------")
    print("오류: 이 스크립트는 모듈로 실행해야 합니다.")
    print("프로젝트 루트(kook_backend) 폴더에서")
    print("\n  python -m scripts.embed_pdfs\n")
    print("---------------------------------------------------------------")
    sys.exit(1)

# --- 1. 설정 ---
PDF_DATA_DIR = 'data/pdf_guidelines' # 👈 (가정) 네 PDF 4개가 이 폴더 안에 있어야 함
VECTOR_STORE_PATH = 'vector_store/faiss_index' # 👈 생성될 로컬 벡터 DB 저장 경로

# --- 2. Bedrock 임베딩 클라이언트 초기화 (LangChain 방식) ---
try:
    # (로컬용) .env의 Access Key/Secret Key를 명시적으로 사용
    bedrock_boto_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.AWS_DEFAULT_REGION,
    )
    
    # LangChain의 BedrockEmbeddings 래퍼 사용
    bedrock_embeddings = BedrockEmbeddings(
        client=bedrock_boto_client,
        model_id="amazon.titan-embed-text-v1" # 👈 (주의) 임베딩용 모델 ID
    )
    print("[Embeddings] Bedrock 임베딩 클라이언트 초기화 성공.")

except Exception as e:
    print(f"[Embeddings] Bedrock 클라이언트 초기화 실패: {e}")
    sys.exit(1)

# --- 3. 메인 실행 로직 ---
def create_vector_store():
    print(f"'{PDF_DATA_DIR}' 폴더에서 PDF 로드를 시작합니다...")
    
    # 1. PDF 로드 (Load)
    # (PyPDFLoader가 기본. pdfplumber를 쓰려면 pip install pdfplumber)
    loader = DirectoryLoader(
        PDF_DATA_DIR, 
        glob="**/*.pdf",    # 이 폴더의 모든 PDF
        loader_cls=PyPDFLoader # PDF 로더 지정
    )
    documents = loader.load()
    
    if not documents:
        print(f"오류: '{PDF_DATA_DIR}'에서 PDF 파일을 찾을 수 없습니다.")
        return

    print(f"PDF {len(documents)}페이지 로드 완료. 텍스트 분할(Chunking) 시작...")

    # 2. 텍스트 분할 (Chunk)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # 1000 글자 단위로
        chunk_overlap=100  # 100 글자씩 겹치게
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"총 {len(chunks)}개의 문단(Chunks)으로 분할 완료.")
    print("Bedrock 임베딩 및 FAISS 벡터 스토어 생성을 시작합니다... (시간이 걸릴 수 있음)")

    try:
        # 3. 임베딩 & 벡터 스토어 생성 (Embed & Store)
        # (이 과정에서 chunks 갯수만큼 Bedrock API가 호출됨)
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=bedrock_embeddings
        )
        
        # 4. 로컬 파일로 저장
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        vector_store.save_local(VECTOR_STORE_PATH)
        
        print(f"\n✅ Vector Store 생성 성공!")
        print(f"'{VECTOR_STORE_PATH}' 폴더에 저장되었습니다.")
        
    except Exception as e:
        print(f"\n❌ Vector Store 생성 실패: {e}")
        print("(Bedrock Titan Embedding 모델 권한이 IAM User에게 있는지 확인하세요.)")

if __name__ == "__main__":
    create_vector_store()