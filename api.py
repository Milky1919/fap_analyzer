# -*- coding: utf-8 -*-
import uvicorn
import sqlite3
import datetime
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

import db_utils
import config

# --- Pydanticモデル定義 ---

class Post(BaseModel):
    # postsテーブルの全カラム
    post_id: int
    post_datetime: str
    title: str
    purpose: str
    original_text: str
    author_name: str
    author_real_age: str
    author_real_gender: str
    author_char_race: str
    author_char_gender: str
    author_char_job: str
    server: str
    voice_chat: str
    server_transfer: str
    sub_char_ok: int

    # evaluation_scoresテーブルから
    unique_score: float

    # tagsテーブルから（後処理で追加）
    tags: Dict[str, List[str]]


# --- FastAPIアプリケーション ---
app = FastAPI()

# --- APIエンドポイント ---

@app.get("/api/v1/posts", response_model=List[Post])
def search_posts(
    # スコア・時間
    min_score: float = None,
    max_age_hours: int = None,
    # プロフィール（完全一致）
    purpose: str = None,
    author_real_gender: str = None,
    author_real_age: str = None,
    author_char_race: str = None,
    server: str = None,
    voice_chat: str = None,
    server_transfer: str = None,
    sub_char_ok: bool = None,
    # タグ・希望条件（部分一致）
    include_wish_real_ages: Optional[List[str]] = Query(default=None),
    include_wish_jobs: Optional[List[str]] = Query(default=None),
    include_playstyle_tags: Optional[List[str]] = Query(default=None),
    include_activity_times: Optional[List[str]] = Query(default=None),
    include_wish_races: Optional[List[str]] = Query(default=None),
    include_wish_char_genders: Optional[List[str]] = Query(default=None),
    include_wish_real_genders: Optional[List[str]] = Query(default=None),
    include_external_tools: Optional[List[str]] = Query(default=None),
):
    conn = None
    try:
        conn = db_utils.setup_database(config.DB_NAME)
        conn.row_factory = sqlite3.Row

        # --- SQLクエリの動的構築 ---
        sql_query = """
            SELECT p.*, es.unique_score
            FROM posts p
            JOIN evaluation_scores es ON p.post_id = es.post_id
        """
        where_clauses = []
        params = {}

        # 1. スコア・時間フィルタ
        if min_score is not None:
            where_clauses.append("es.unique_score >= :min_score")
            params["min_score"] = min_score
        if max_age_hours is not None:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)
            where_clauses.append("p.post_datetime >= :cutoff_time")
            params["cutoff_time"] = cutoff_time.strftime('%Y/%m/%d %H:%M')

        # 2. プロフィールフィルタ (完全一致)
        profile_filters = {
            "purpose": purpose,
            "author_real_gender": author_real_gender,
            "author_real_age": author_real_age,
            "author_char_race": author_char_race,
            "server": server,
            "voice_chat": voice_chat,
            "server_transfer": server_transfer,
            "sub_char_ok": 1 if sub_char_ok is True else (0 if sub_char_ok is False else None),
        }
        for key, value in profile_filters.items():
            if value is not None:
                where_clauses.append(f"p.{key} = :{key}")
                params[key] = value

        # 3. タグフィルタ (部分一致)
        tag_filters = {
            "wish_real_ages": include_wish_real_ages,
            "wish_jobs": include_wish_jobs,
            "playstyle_tags": include_playstyle_tags,
            "activity_times": include_activity_times,
            "wish_races": include_wish_races,
            "wish_char_genders": include_wish_char_genders,
            "wish_real_genders": include_wish_real_genders,
            "external_tools": include_external_tools,
        }
        for category, tags in tag_filters.items():
            if tags:
                # プレースホルダーを動的に生成 (例: wish_jobs_0, wish_jobs_1)
                tag_placeholders = [f":{category}_{i}" for i in range(len(tags))]
                for i, tag_name in enumerate(tags):
                    params[f"{category}_{i}"] = tag_name

                where_clauses.append(f"""
                    EXISTS (
                        SELECT 1 FROM post_tags pt
                        JOIN tags t ON pt.tag_id = t.tag_id
                        WHERE pt.post_id = p.post_id
                          AND t.tag_category = '{category}'
                          AND t.tag_name IN ({', '.join(tag_placeholders)})
                    )
                """)

        # --- クエリの結合と実行 ---
        if where_clauses:
            sql_query += " WHERE " + " AND ".join(where_clauses)

        sql_query += " ORDER BY p.post_datetime DESC"

        cursor = conn.cursor()
        cursor.execute(sql_query, params)
        posts_rows = cursor.fetchall()

        if not posts_rows:
            return []

        # --- N+1問題対策: タグを一括取得 ---
        post_ids = [row['post_id'] for row in posts_rows]

        tags_sql = f"""
            SELECT pt.post_id, t.tag_category, t.tag_name
            FROM post_tags pt
            JOIN tags t ON pt.tag_id = t.tag_id
            WHERE pt.post_id IN ({','.join('?' * len(post_ids))})
        """
        cursor.execute(tags_sql, post_ids)
        tags_rows = cursor.fetchall()

        # --- 投稿ごとにタグをマッピング ---
        tags_by_post_id = {pid: {} for pid in post_ids}
        for row in tags_rows:
            pid = row['post_id']
            category = row['tag_category']
            if category not in tags_by_post_id[pid]:
                tags_by_post_id[pid][category] = []
            tags_by_post_id[pid][category].append(row['tag_name'])

        # --- 最終的なレスポンスを構築 ---
        response_data = []
        for row in posts_rows:
            post_dict = dict(row)
            post_dict['tags'] = tags_by_post_id.get(post_dict['post_id'], {})
            response_data.append(Post(**post_dict))

        return response_data

    except Exception as e:
        # 実際のアプリケーションでは、より詳細なロギングを行うべき
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


# --- サーバー起動設定 ---

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
