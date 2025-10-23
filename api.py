# -*- coding: utf-8 -*-
import uvicorn
import sqlite3
import datetime
import json
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
    score_breakdown: str
    is_repost: int
    penalty: float
    max_similarity_score: float

    # tagsテーブルから（後処理で追加）
    tags: Dict[str, List[str]]

    # APIで動的に生成
    score_reasoning: str
    score_summary_list: List[str]


# --- ヘルパー関数 ---

def format_score_reasoning(score_breakdown_json: str) -> str:
    """
    score_breakdownのJSON文字列を人間が読める形式の日本語テキストに変換する。
    """
    if not score_breakdown_json:
        return "スコア内訳データがありません。"

    try:
        data = json.loads(score_breakdown_json)
    except json.JSONDecodeError:
        return "スコア内訳の解析に失敗しました。"

    if data.get("reason") == "No candidates":
        return "比較対象となる類似投稿が見つかりませんでした。"

    reasoning_parts = []

    # 静的プロファイル
    static_score = data.get("static", 0)
    reasoning_parts.append(f"静的プロファイル: {static_score:.1f} / {config.STATIC_PROFILE_SCORE_MAX}点")

    # 行動・嗜好
    behavioral_score = data.get("behavioral", 0)
    reasoning_parts.append(f"行動・嗜好パターン: {behavioral_score:.1f} / {config.BEHAVIORAL_PATTERN_SCORE_MAX}点")

    # 言語的指紋 (意味内容 + 文体)
    semantic_score = data.get("semantic", 0)
    stylistic_score = data.get("stylistic", 0)
    linguistic_total = semantic_score + stylistic_score
    reasoning_parts.append(f"言語的指紋: {linguistic_total:.1f} / {config.LINGUISTIC_FINGERPRINT_SCORE_MAX}点")

    # 一貫性ボーナス
    bonus = data.get("bonus", 0)
    if bonus > 0:
        reasoning_parts.append(f"一貫性ボーナス: +{bonus}点")

    return "\n".join(reasoning_parts)


def generate_score_summary_list(
    is_repost: int,
    penalty: float,
    max_similarity_score: float,
    score_breakdown_json: str
) -> List[str]:
    """
    評価データに基づいて、スコアに関する総評のリストを生成する。
    """
    summary_list = []

    try:
        breakdown = json.loads(score_breakdown_json) if score_breakdown_json else {}
    except json.JSONDecodeError:
        breakdown = {}

    # ルール1: 再投稿ペナルティ
    if is_repost == 1 and penalty < 0:
        summary_list.append("高頻度または酷似した再投稿と判定され、ペナルティが適用されています。")

    # ルール2: 静的プロファイルの類似性
    static_score = breakdown.get("static", 0)
    if static_score >= config.REPOST_STATIC_SCORE_THRESHOLD:
        summary_list.append("過去の投稿とプロフィール（年代、性別、種族など）が酷似しています。")

    # ルール3: 内容と文体の乖離
    semantic_score = breakdown.get("semantic", 0)
    stylistic_score = breakdown.get("stylistic", 0)
    if semantic_score >= 13 and stylistic_score <= 7:
        summary_list.append("内容は酷似していますが、文体（書き方）が異なるため、別人の可能性も考慮されました。")

    # デフォルトメッセージ
    if not summary_list:
        if max_similarity_score < config.REPOST_THRESHOLD:
            summary_list.append("ユニーク性の高い新規投稿です。")
        else:
            summary_list.append("過去の投稿と類似点が見られますが、再投稿とは判定されませんでした。")


    return summary_list


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
            SELECT
                p.*,
                es.unique_score,
                es.score_breakdown,
                es.is_repost,
                es.penalty,
                es.max_similarity_score
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
                search_tags = set(tags)
                unspecified_tags = set(config.UNSPECIFIED_TAG_MAP.get(category, []))

                # 具体的なタグが指定され、かつそれが「指定なし」タグのみでない場合に、「指定なし」タグを検索条件に加える
                if not search_tags.issubset(unspecified_tags):
                    search_tags.update(unspecified_tags)

                final_tags = list(search_tags)

                # プレースホルダーを動的に生成 (例: wish_jobs_0, wish_jobs_1)
                tag_placeholders = [f":{category}_{i}" for i in range(len(final_tags))]
                for i, tag_name in enumerate(final_tags):
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

            # タグ情報を追加
            post_dict['tags'] = tags_by_post_id.get(post_dict['post_id'], {})

            # スコアの理由とサマリーを生成
            post_dict['score_reasoning'] = format_score_reasoning(row['score_breakdown'])
            post_dict['score_summary_list'] = generate_score_summary_list(
                is_repost=row['is_repost'],
                penalty=row['penalty'],
                max_similarity_score=row['max_similarity_score'],
                score_breakdown_json=row['score_breakdown']
            )

            response_data.append(Post(**post_dict))

        return response_data

    except Exception as e:
        # 実際のアプリケーションでは、より詳細なロギングを行うべき
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


# --- サーバー起動設定 ---

# if __name__ == "__main__":
#     uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
