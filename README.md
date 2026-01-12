# K-OOK : 현지 식재료로 쉽게 만드는 K-Food 레시피 앱
## 2025 숙명여대 캠퍼스타운-AWS 해커톤 참여작
## 💡 Key Features & Data Logic

### 1. 🛒 Commerce Data 기반 맞춤 레시피
현지 식료품점 판매 데이터를 분석하여, 외국인들이 실제로 집에 가지고 있을 법한 재료로 한식을 추천합니다.
- **데이터 분석:** Grocery Sales Database(6개 파일 병합)를 분석하여 판매량 상위 식재료(Beef, Wine, Cheese, Bread 등) 도출.
- **Hot Ingredients:** 판매량 데이터를 기반으로 메인 화면에 '인기 재료'를 노출하고, 해당 재료로 만들 수 있는 퓨전/정통 한식 레시피 매칭.

### 2. 📈 Reddit 트렌드 분석 (Trending Menu)
해외 커뮤니티에서 실제로 유행하는 K-푸드를 실시간으로 파악하여 제안합니다.
- **데이터 수집:** Reddit의 `r/koreanFood` 서브레딧에서 최근 게시물 1,000개 수집.
- **N-gram 분석:** 텍스트 마이닝(Uni/Bi/Tri-gram)을 통해 'toasted sesame seeds', 'kimchi fried rice', 'black bean noodles' 등 상위 빈도 메뉴 추출.
- **Top 15 선정:** 빈도수 상위 15개 메뉴를 트렌딩 메뉴로 선정하여 메인 홈에 큐레이션.

### 3. 🤖 RAG 기반 레시피 챗봇 (Chef's Tip)
단순한 레시피 생성을 넘어, 신뢰할 수 있고 맛있는 조언을 제공합니다.
- **RAG (검색 증강 생성):** AWS Knowledge Base에 검증된 레시피 PDF 데이터를 벡터화하여 저장. LLM이 이를 참조하여 환각(Hallucination) 없이 정확한 조리법을 안내.
- **프롬프트 엔지니어링:**
  - **괴식 방지:** 딸기 김치 라떼와 같은 이질적인 조합을 추천하지 않도록 파라미터 튜닝 (`top-p=0.7`, `temperature=0.3`).
  - **Chef's Tip:** "감자는 너무 작게 썰지 마세요", "볶음밥엔 단무지가 잘 어울립니다" 등 셰프의 꿀팁(Tip) 섹션을 별도로 생성하도록 프롬프트 구조화.

## 📐 Architecture
```mermaid
flowchart LR
    A["Client (React)"] <--> B["Server (FastAPI)"]
    B <--> C[("SQLite DB")]
    B <--> D["Data Analysis (Pandas)"]
    B <--> E["RAG Pipeline"]
    E --> F["LangChain"]
    F --> G["AWS Bedrock"]
    F --> H[("AWS Knowledge Base (S3)")]
