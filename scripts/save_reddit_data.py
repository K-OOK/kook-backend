from datetime import datetime
import csv
import dotenv
import praw
import os

dotenv.load_dotenv()

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_PASSWORD = os.getenv('REDDIT_PASSWORD')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
REDDIT_USERNAME = os.getenv('REDDIT_USERNAME')

# 1. PRAW 인스턴스 생성 (아이디/비밀번호/앱 정보만 주면 됨)
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    password=REDDIT_PASSWORD,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
)

# CSV 파일명
csv_filename = 'data/reddit_koreanfood.csv'

try:
    # 2. 검색 실행 (PRAW가 내부적으로 토큰 처리)
    search_results = reddit.subreddit('koreanfood').new(limit=3000)

    # 3. 데이터 수집
    data_list = []
    for submission in search_results:
        created_date = datetime.utcfromtimestamp(submission.created_utc)
        data_list.append({
            'title': submission.title,
            'subreddit': submission.subreddit.display_name,
            'score': submission.score,
            'content': submission.selftext,
            'created_date': created_date.strftime('%Y-%m-%d %H:%M:%S'),
            'URL': f'https://www.reddit.com{submission.permalink}',
            'num_comments': submission.num_comments,
            'upvote_ratio': submission.upvote_ratio
        })

    # 4. CSV 파일로 저장
    if data_list:
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['title', 'subreddit', 'score', 'content', 'created_date', 'URL', 'num_comments', 'upvote_ratio']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(data_list)
        
        print(f"총 {len(data_list)}개의 데이터를 {csv_filename} 파일로 저장했습니다.")
    else:
        print("저장할 데이터가 없습니다.")

except Exception as e:
    print(f"PRAW 실행 중 오류 발생: {e}")