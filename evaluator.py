# -*- coding: utf-8 -*-
import sqlite3
import datetime
import json
import pickle
import logging
import re
import unicodedata
import argparse
import traceback
from typing import List, Dict, Any, Optional, Set

import MeCab
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 共有モジュールをインポート
import db_utils
import config

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- データ構造定義 ---

class Post:
    """
    postsテーブルの1レコードを表現するクラス。
    関連するタグはDBからJOINして取得する。
    """
    def __init__(self, conn: sqlite3.Connection, db_row: Dict[str, Any]):
        # 基本情報のマッピング
        self.post_id: int = db_row['post_id']
        dt_str = db_row.get('post_datetime')
        self.post_datetime: Optional[datetime.datetime] = datetime.datetime.strptime(dt_str, '%Y/%m/%d %H:%M') if dt_str else None
        self.title: str = db_row['title']
        self.purpose: str = db_row['purpose']
        self.original_text: str = db_row['original_text']
        self.author_name: str = db_row['author_name']
        age_str = (db_row['author_real_age'] or '0').replace('代', '').replace('？', '0')
        self.author_real_age: int = int(age_str) if age_str.isdigit() else 0
        self.author_real_gender: str = db_row['author_real_gender']
        self.author_char_race: str = db_row['author_char_race']
        self.author_char_gender: str = db_row['author_char_gender']
        self.author_char_job: str = db_row['author_char_job']
        self.server: str = db_row['server']
        self.voice_chat: str = db_row['voice_chat']
        self.server_transfer: str = db_row['server_transfer']
        self.sub_char_ok: bool = bool(db_row['sub_char_ok'])

        # タグ情報をDBから取得
        self.tags_by_category: Dict[str, List[str]] = self._load_tags(conn)

        # 全タグ集合を生成
        self.all_tags: Set[str] = self._create_all_tags_set()

    def _load_tags(self, conn: sqlite3.Connection) -> Dict[str, List[str]]:
        """post_idに紐づくタグをDBから取得し、カテゴリ別に分類して返す"""
        cursor = conn.cursor()
        query = """
            SELECT t.tag_category, t.tag_name
            FROM post_tags pt
            JOIN tags t ON pt.tag_id = t.tag_id
            WHERE pt.post_id = ?
        """
        cursor.execute(query, (self.post_id,))
        tags = {}
        for row in cursor.fetchall():
            category, name = row['tag_category'], row['tag_name']
            if category not in tags:
                tags[category] = []
            tags[category].append(name)
        return tags

    def _create_all_tags_set(self) -> Set[str]:
        """行動・嗜好パターン分析用の全タグ集合を生成する"""
        tags = set()
        for category_tags in self.tags_by_category.values():
            tags.update(category_tags)

        # タグとして扱う他の属性も追加
        if self.voice_chat:
            tags.add(f"VC:{self.voice_chat}")
        if self.server_transfer:
            tags.add(self.server_transfer)
        if self.sub_char_ok:
            tags.add("サブキャラ可")
        return tags

    def __repr__(self) -> str:
        return f"<Post id={self.post_id} name='{self.author_name}'>"

# --- データベース関連 (評価対象取得) ---

def get_posts_to_evaluate(conn: sqlite3.Connection, args: argparse.Namespace) -> List[Post]:
    """コマンドライン引数に基づいて評価対象の投稿リストを取得する"""
    cursor = conn.cursor()
    if args.post_id:
        logging.info(f"--post-idが指定されました。投稿ID: {args.post_id} を評価します。")
        cursor.execute("SELECT * FROM posts WHERE post_id = ?", (args.post_id,))
    elif args.re_evaluate_all:
        logging.info("--re-evaluate-allが指定されました。全投稿を再評価します。")
        cursor.execute("DELETE FROM evaluation_scores")
        conn.commit()
        logging.info("既存の評価スコアをすべて削除しました。")
        cursor.execute("SELECT * FROM posts ORDER BY post_datetime ASC")
    else:
        logging.info("未評価の投稿をすべて評価します。（デフォルトモード）")
        cursor.execute("""
            SELECT p.* FROM posts p
            LEFT JOIN evaluation_scores es ON p.post_id = es.post_id
            WHERE es.post_id IS NULL
            ORDER BY p.post_datetime ASC
        """)
    rows = cursor.fetchall()
    if not rows:
        logging.info("評価対象となる投稿はありませんでした。")
        return []

    # Postオブジェクトの生成にはDB接続が必要
    target_posts = [Post(conn, dict(row)) for row in rows]
    logging.info(f"{len(target_posts)} 件の投稿を評価対象として取得しました。")
    return target_posts

def get_all_original_texts(conn: sqlite3.Connection) -> List[str]:
    """postsテーブルから全てのoriginal_textを取得する"""
    cursor = conn.cursor()
    cursor.execute("SELECT original_text FROM posts ORDER BY post_datetime ASC")
    return [row['original_text'] for row in cursor.fetchall() if row['original_text']]

# --- テキスト処理関連 ---
def normalize_text(text: str) -> str:
    """テキストを正規化する"""
    if not text: return ""
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = re.sub(r'[、。！？]', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_with_mecab(text: str) -> str:
    """MeCabでわかち書き"""
    tagger = MeCab.Tagger("-Owakati -r /etc/mecabrc")
    return tagger.parse(text).strip()

def prepare_vectorizer(conn: sqlite3.Connection) -> Optional[TfidfVectorizer]:
    """TF-IDFベクトル化器を準備"""
    try:
        with open(config.VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        logging.info(f"ベクトル化器を '{config.VECTORIZER_PATH}' から読み込みました。")
        return vectorizer
    except FileNotFoundError:
        logging.info("ベクトル化器ファイルが見つかりません。新規に生成します。")
        corpus = get_all_original_texts(conn)
        if not corpus:
            logging.error("コーパスの取得に失敗しました。")
            return None
        normalized_corpus = [normalize_text(text) for text in corpus]
        stop_words = ['です', 'ます', 'の', 'は', 'が', 'を', 'に', 'と', 'も', 'で', 'いる', 'する', 'ある']
        vectorizer = TfidfVectorizer(tokenizer=tokenize_with_mecab, stop_words=stop_words)
        vectorizer.fit(normalized_corpus)
        with open(config.VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        logging.info(f"学習済みのベクトル化器を '{config.VECTORIZER_PATH}' に保存しました。")
        return vectorizer
    except Exception as e:
        logging.error(f"ベクトル化器の準備中にエラー: {e}")
        return None

# --- 評価ロジック ---

def get_candidate_posts(conn: sqlite3.Connection, target_post: Post) -> List[Post]:
    """高速フィルタリングで候補投稿を取得"""
    cursor = conn.cursor()
    try:
        query = """
            SELECT * FROM posts
            WHERE author_real_gender = ? AND author_real_age = ? AND author_char_race = ?
              AND author_char_gender = ? AND post_id != ? AND post_datetime < ?
        """
        params = (
            target_post.author_real_gender,
            f"{target_post.author_real_age}代" if target_post.author_real_age != 0 else "？代",
            target_post.author_char_race,
            target_post.author_char_gender,
            target_post.post_id,
            target_post.post_datetime.strftime('%Y/%m/%d %H:%M') if target_post.post_datetime else ''
        )
        cursor.execute(query, params)
        rows = cursor.fetchall()
        candidate_posts = [Post(conn, dict(row)) for row in rows]
        logging.info(f"高速フィルタリングにより {len(candidate_posts)} 件の比較候補を検出しました。")
        return candidate_posts
    except sqlite3.Error as e:
        logging.error(f"候補投稿の取得中にDBエラー: {e}")
        return []

def save_evaluation_score(conn: sqlite3.Connection, post_id: int, result: Dict[str, Any]):
    """評価結果をDBに保存"""
    cursor = conn.cursor()
    try:
        query = """
            INSERT OR REPLACE INTO evaluation_scores (
                post_id, evaluation_datetime, unique_score, most_similar_post_id,
                max_similarity_score, is_repost, penalty, score_breakdown, author_post_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            post_id, datetime.datetime.now().isoformat(), result.get('unique_score'),
            result.get('most_similar_post_id'), result.get('max_similarity_score'),
            result.get('is_repost'), result.get('penalty'),
            json.dumps(result.get('score_breakdown', {}), ensure_ascii=False),
            result.get('author_post_count')
        )
        cursor.execute(query, params)
        conn.commit()
        logging.info(f"Post ID {post_id} の評価結果をDBに保存しました。")
    except sqlite3.Error as e:
        logging.error(f"評価結果の保存中にDBエラー: {e}")

def print_summary(target_post: Post, result: Dict[str, Any]):
    """評価結果サマリーをコンソール出力"""
    print("\n" + "="*60 + f"\n{'評価結果サマリー':^60}\n" + "="*60)
    print(f" 評価対象投稿ID: {target_post.post_id}")
    print(f" 投稿者名:         {target_post.author_name}")
    print(f" 投稿日時:         {target_post.post_datetime.strftime('%Y-%m-%d %H:%M') if target_post.post_datetime else 'N/A'}")
    print("-"*60)
    print(f" 【最終ユニークスコア】: {result['unique_score']:.2f} / 100")
    print("-"*60)
    print(f"   - 最大類似度スコア:     {result['max_similarity_score']:.2f}")
    print(f"   - 最も類似した投稿ID: {result.get('most_similar_post_id', 'N/A')}")
    print(f"   - 再投稿判定:           {'はい' if result['is_repost'] else 'いいえ'}")
    print(f"   - 適用ペナルティ:       {result['penalty']:.2f}")
    print("="*60 + "\n")

def calculate_static_profile_score(post_a: Post, post_b: Post) -> float:
    """A) 静的プロファイル類似度スコア"""
    score = 0
    age_diff = abs(post_a.author_real_age - post_b.author_real_age)
    score += config.REAL_AGE_SCORE_MAX * (1 - min(5, age_diff) / 5)
    if post_a.author_char_race == post_b.author_char_race and post_a.author_char_gender == post_b.author_char_gender:
        score += config.CHAR_INFO_SCORE_MAX
    if post_a.author_char_job == post_b.author_char_job and post_a.author_name == post_b.author_name:
        score += config.VARIABLE_PROFILE_SCORE_MAX
    return score

def calculate_behavioral_pattern_score(post_a: Post, post_b: Post, tag_weights: Dict[str, float]) -> float:
    """B) 行動・嗜好パターン類似度スコア"""
    intersection = post_a.all_tags.intersection(post_b.all_tags)
    union = post_a.all_tags.union(post_b.all_tags)
    if not union: return 0.0
    intersection_weight = sum(tag_weights.get(tag, 0) for tag in intersection)
    union_weight = sum(tag_weights.get(tag, 0) for tag in union)
    return config.BEHAVIORAL_PATTERN_SCORE_MAX * (intersection_weight / union_weight) if union_weight > 0 else 0.0

def _calculate_tag_weights_from_db(conn: sqlite3.Connection) -> Dict[str, float]:
    """DBから全タグの出現回数を元に、タグの重みを計算する（ヘルパー関数）"""
    cursor = conn.cursor()
    tag_counts = {}

    # 1. 'tags'テーブルに由来するタグの出現回数を効率的に一括で計算
    query = """
        SELECT t.tag_name, COUNT(pt.post_id) as count
        FROM tags t
        JOIN post_tags pt ON t.tag_id = pt.tag_id
        GROUP BY t.tag_name;
    """
    cursor.execute(query)
    for row in cursor.fetchall():
        tag_counts[row['tag_name']] = row['count']

    # 2. 'posts'テーブルの属性に由来するタグの出現回数を計算
    # (voice_chat, server_transfer, sub_char_ok)
    cursor.execute("SELECT voice_chat, server_transfer, sub_char_ok FROM posts")
    posts_attributes = cursor.fetchall()

    for row in posts_attributes:
        # Voice Chat
        if row['voice_chat']:
            vc_tag = f"VC:{row['voice_chat']}"
            tag_counts[vc_tag] = tag_counts.get(vc_tag, 0) + 1
        # Server Transfer
        if row['server_transfer']:
            st_tag = row['server_transfer']
            tag_counts[st_tag] = tag_counts.get(st_tag, 0) + 1
        # Sub Char OK
        if bool(row['sub_char_ok']):
            sc_tag = "サブキャラ可"
            tag_counts[sc_tag] = tag_counts.get(sc_tag, 0) + 1

    # 3. 重みを計算: w(t) = 1 / np.log(1 + (出現回数))
    tag_weights = {tag: 1 / np.log(1 + count) for tag, count in tag_counts.items() if count > 0}
    logging.info(f"{len(tag_weights)}個のタグの重みを計算しました。")
    return tag_weights

def prepare_tag_weights(conn: sqlite3.Connection) -> Dict[str, float]:
    """タグの重みデータを準備"""
    try:
        with open(config.TAG_WEIGHTS_PATH, 'rb') as f:
            tag_weights = pickle.load(f)
        logging.info(f"タグ重みデータを '{config.TAG_WEIGHTS_PATH}' から読み込みました。")
        return tag_weights
    except FileNotFoundError:
        logging.info("タグ重みデータファイルが見つかりません。新規に生成します。")
        tag_weights = _calculate_tag_weights_from_db(conn)
        with open(config.TAG_WEIGHTS_PATH, 'wb') as f:
            pickle.dump(tag_weights, f)
        logging.info(f"タグ重みデータを '{config.TAG_WEIGHTS_PATH}' に保存しました。")
        return tag_weights
    except Exception as e:
        logging.error(f"タグ重みデータの準備中にエラー: {e}")
        return {}

def create_stylistic_feature_vector(text: str) -> np.ndarray:
    """文体特徴ベクトルを生成"""
    if not text: return np.zeros(8)
    text_len = len(text)
    features = [
        len(re.findall(r'[！-～]', text)) / (len(re.findall(r'[!-~]', text)) + len(re.findall(r'[！-～]', text)) or 1),
        len(re.findall(r'[！？w♪✨🍀🎀🐰🧸💌]', text)) / text_len,
        len(re.findall(r'([。、])\s', text)) / (len(re.findall(r'[。、]', text)) or 1),
        text.count('\n') / text_len,
        len(re.findall(r'[\u3040-\u309F]', text)) / text_len,
        len(re.findall(r'[\u30A0-\u30FF]', text)) / text_len,
        len(re.findall(r'[\u4E00-\u9FFF]', text)) / text_len,
        0.0
    ]
    return np.array(features)

def calculate_linguistic_fingerprint_score(post_a: Post, post_b: Post, vectorizer: TfidfVectorizer) -> Dict[str, float]:
    """C) 言語的指紋類似度スコア"""
    texts = [normalize_text(post_a.original_text), normalize_text(post_b.original_text)]
    tfidf_vectors = vectorizer.transform(texts)
    semantic_sim = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors[1:2])[0][0]
    semantic_score = config.SEMANTIC_SCORE_MAX * semantic_sim
    style_vec_a = create_stylistic_feature_vector(post_a.original_text)
    style_vec_b = create_stylistic_feature_vector(post_b.original_text)
    euclidean_dist = np.linalg.norm(style_vec_a - style_vec_b)
    stylistic_sim = np.exp(-0.5 * euclidean_dist)
    stylistic_score = config.STYLISTIC_SCORE_MAX * stylistic_sim
    reliability = min(1.0, len(post_a.original_text) / config.RELIABILITY_TEXT_LENGTH if post_a.original_text else 0.0)
    final_score = (semantic_score + stylistic_score) * reliability
    return {"total": final_score, "semantic": semantic_score, "stylistic": stylistic_score}

def trace_repost_chain(conn: sqlite3.Connection, start_post_id: int, visited: Set[int]) -> int:
    """再投稿の連鎖を遡る"""
    if start_post_id is None or start_post_id in visited: return 0
    visited.add(start_post_id)
    cursor = conn.cursor()
    cursor.execute("SELECT most_similar_post_id, is_repost FROM evaluation_scores WHERE post_id = ?", (start_post_id,))
    result = cursor.fetchone()
    if not result or not result['is_repost']: return 1
    return 1 + trace_repost_chain(conn, result['most_similar_post_id'], visited)

def get_post_by_id(conn: sqlite3.Connection, post_id: int) -> Optional[Post]:
    """指定されたpost_idのPostオブジェクトを取得する"""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM posts WHERE post_id = ?", (post_id,))
    row = cursor.fetchone()
    return Post(conn, dict(row)) if row else None

def evaluate_post(target_post: Post, candidates: List[Post], vectorizer: TfidfVectorizer, tag_weights: Dict[str, float], conn: sqlite3.Connection) -> Dict[str, Any]:
    """1件の投稿を評価（ペナルティロジック改善版）"""
    if not candidates:
        return {"unique_score": 100.0, "most_similar_post_id": None, "max_similarity_score": 0.0,
                "is_repost": 0, "penalty": 0.0, "author_post_count": 1, "score_breakdown": {"reason": "No candidates"}}

    max_sim_score, most_sim_post_id, best_scores = 0.0, None, {}
    most_similar_post_obj = None # 猶予期間チェックのためにPostオブジェクト自体も保持

    for cand_post in candidates:
        static = calculate_static_profile_score(target_post, cand_post)
        behavioral = calculate_behavioral_pattern_score(target_post, cand_post, tag_weights)
        linguistic = calculate_linguistic_fingerprint_score(target_post, cand_post, vectorizer)
        base_sim = static + behavioral + linguistic["total"]
        bonus = config.CONSISTENCY_BONUS if target_post.server == cand_post.server else 0
        current_sim = base_sim + bonus
        if current_sim > max_sim_score:
            max_sim_score = current_sim
            most_sim_post_id = cand_post.post_id
            most_similar_post_obj = cand_post # オブジェクトを更新
            best_scores = {"static": static, "behavioral": behavioral, **linguistic, "bonus": bonus}

    is_repost = 1 if max_sim_score >= config.REPOST_THRESHOLD and best_scores.get("static", 0) >= config.REPOST_STATIC_SCORE_THRESHOLD else 0
    post_count = 1 + trace_repost_chain(conn, most_sim_post_id, set()) if is_repost and most_sim_post_id else 1
    penalty = 0.0

    if is_repost and target_post.post_datetime and most_similar_post_obj and most_similar_post_obj.post_datetime:
        # 1. 猶予期間（マージン）のチェック
        time_diff = target_post.post_datetime - most_similar_post_obj.post_datetime
        if time_diff.total_seconds() / 60 < config.REPOST_GRACE_PERIOD_MINUTES:
            penalty = 0.0
            logging.info(f"再投稿猶予期間内のため、ペナルティを 0 に設定します。(時間差: {time_diff})")
        else:
            # 2. 新しいペナルティ回復曲線の計算
            N = post_count
            # Dは「類似投稿からの」経過日数ではなく、「評価実行時点からの」経過日数
            D = (datetime.datetime.now() - target_post.post_datetime).days

            base_penalty = config.BASE_PENALTY_COEFFICIENT - (config.PENALTY_PER_REPOST * (N - 1))

            recovery = 0.0
            if D > config.NO_RECOVERY_PERIOD_DAYS:
                # 回復期間に入っている場合のみ回復量を計算
                recovery_days = D - config.NO_RECOVERY_PERIOD_DAYS
                recovery_period = config.DAYS_FOR_FULL_RECOVERY - config.NO_RECOVERY_PERIOD_DAYS
                recovery_rate = min(1.0, recovery_days / recovery_period) if recovery_period > 0 else 1.0

                recovery_limit = max(config.MIN_RECOVERY_LIMIT, config.MAX_RECOVERY_LIMIT - (N - 2))
                recovery = recovery_rate * recovery_limit

            penalty = base_penalty + recovery

    unique_score = max(0, min(100, 100 - max_sim_score + penalty))
    return {"unique_score": unique_score, "most_similar_post_id": most_sim_post_id, "max_similarity_score": max_sim_score,
            "is_repost": is_repost, "penalty": penalty, "author_post_count": post_count, "score_breakdown": best_scores}

# --- メイン処理 ---
def main():
    """評価プログラムのメイン処理"""
    parser = argparse.ArgumentParser(description="投稿のユニークスコアを評価するプログラム")
    parser.add_argument('--post-id', type=int, help='指定したpost_idの投稿のみを評価する')
    parser.add_argument('--re-evaluate-all', action='store_true', help='すべての投稿を強制的に再評価する')
    args = parser.parse_args()
    logging.info("--- 投稿評価プログラムを開始します ---")
    conn = None
    try:
        conn = db_utils.setup_database(config.DB_NAME)
        conn.row_factory = sqlite3.Row
        # 評価結果テーブルのカラム追加・インデックス作成もsetup_databaseに含めるべきだが、
        # 後方互換性のため、元のevaluator.pyにあった処理をここに残す
        setup_evaluation_specific_tables(conn)

        target_posts = get_posts_to_evaluate(conn, args)
        if not target_posts:
            logging.info("評価対象が見つからなかったため、処理を終了します。")
            return
        vectorizer = prepare_vectorizer(conn)
        if not vectorizer:
            logging.error("ベクトル化器の準備に失敗したため、処理を中断します。")
            return
        tag_weights = prepare_tag_weights(conn)

        for i, target_post in enumerate(target_posts):
            logging.info(f"--- ({i+1}/{len(target_posts)}) post_id: {target_post.post_id} の評価を開始 ---")
            candidates = get_candidate_posts(conn, target_post)
            result = evaluate_post(target_post, candidates, vectorizer, tag_weights, conn)
            save_evaluation_score(conn, target_post.post_id, result)
            print_summary(target_post, result)
    except Exception as e:
        logging.critical("評価プログラムの実行中に致命的なエラーが発生しました。", exc_info=True)
        if conn:
            db_utils.log_to_db(conn, 'CRITICAL', 'evaluator.py', f"致命的なエラー: {e}", traceback.format_exc())
    finally:
        if conn:
            conn.close()
        logging.info("--- 投稿評価プログラムを終了します ---")

def setup_evaluation_specific_tables(conn: sqlite3.Connection):
    """
    evaluator.py固有のテーブル設定（evaluation_scoresテーブルのカラム追加など）
    """
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluation_scores (
        post_id INTEGER PRIMARY KEY, evaluation_datetime TEXT, unique_score REAL,
        most_similar_post_id INTEGER, max_similarity_score REAL, is_repost INTEGER, penalty REAL
    );
    """)
    cursor.execute("PRAGMA table_info(evaluation_scores);")
    existing_columns = [row['name'] for row in cursor.fetchall()]
    if 'score_breakdown' not in existing_columns:
        cursor.execute("ALTER TABLE evaluation_scores ADD COLUMN score_breakdown TEXT;")
    if 'author_post_count' not in existing_columns:
        cursor.execute("ALTER TABLE evaluation_scores ADD COLUMN author_post_count INTEGER;")
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_post_filter
    ON posts (author_real_gender, author_real_age, author_char_race, author_char_gender);
    """)
    conn.commit()

if __name__ == "__main__":
    main()
