# -*- coding: utf-8 -*-
import sqlite3
import datetime
import json
import pickle
import logging
import re
import unicodedata
import argparse # argparseをインポート
from typing import List, Dict, Any, Optional, Set

import MeCab
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

import config # 設定ファイルをインポート

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- データ構造定義 ---

class Post:
    """
    postsテーブルの1レコードを表現するクラス
    """
    def __init__(self, db_row: Dict[str, Any]):
        self.post_id: int = db_row['post_id']
        # post_datetimeがNoneの場合や空文字列の場合のフォールバックを追加
        dt_str = db_row.get('post_datetime')
        self.post_datetime: Optional[datetime.datetime] = datetime.datetime.strptime(dt_str, '%Y/%m/%d %H:%M') if dt_str else None
        self.title: str = db_row['title']
        self.purpose: str = db_row['purpose']
        self.original_text: str = db_row['original_text']
        self.author_name: str = db_row['author_name']

        # 年代は数値として扱う
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

        # JSON形式の文字列をリスト/セットに変換
        self.external_tools: List[str] = json.loads(db_row['external_tools'] or '[]')
        self.playstyle_tags: List[str] = json.loads(db_row['playstyle_tags'] or '[]')
        self.activity_times: List[str] = json.loads(db_row['activity_times'] or '[]')
        self.wish_races: List[str] = json.loads(db_row['wish_races'] or '[]')
        self.wish_char_genders: List[str] = json.loads(db_row['wish_char_genders'] or '[]')
        self.wish_jobs: List[str] = json.loads(db_row['wish_jobs'] or '[]')
        self.wish_real_genders: List[str] = json.loads(db_row['wish_real_genders'] or '[]')
        self.wish_real_ages: List[str] = json.loads(db_row['wish_real_ages'] or '[]')


        # 全タグ集合を生成
        self.all_tags: Set[str] = self._create_all_tags_set()

    def _create_all_tags_set(self) -> Set[str]:
        """行動・嗜好パターン分析用の全タグ集合を生成する"""
        tags = set()
        tags.update(self.playstyle_tags)
        tags.update(self.activity_times)
        tags.update(self.wish_races)
        tags.update(self.wish_char_genders)
        tags.update(self.wish_jobs)
        tags.update(self.wish_real_genders)
        tags.update(self.wish_real_ages)
        tags.update(self.external_tools)
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
    """
    コマンドライン引数に基づいて評価対象の投稿リストを取得する
    """
    cursor = conn.cursor()

    if args.post_id:
        logging.info(f"--post-idが指定されました。投稿ID: {args.post_id} を評価します。")
        cursor.execute("SELECT * FROM posts WHERE post_id = ?", (args.post_id,))
        rows = cursor.fetchall()
    elif args.re_evaluate_all:
        logging.info("--re-evaluate-allが指定されました。全投稿を再評価します。")
        # 評価テーブルをクリア
        cursor.execute("DELETE FROM evaluation_scores")
        conn.commit()
        logging.info("既存の評価スコアをすべて削除しました。")
        cursor.execute("SELECT * FROM posts ORDER BY post_datetime ASC")
        rows = cursor.fetchall()
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

    target_posts = [Post(dict(row)) for row in rows]
    logging.info(f"{len(target_posts)} 件の投稿を評価対象として取得しました。")
    return target_posts

def get_all_original_texts(conn: sqlite3.Connection) -> List[str]:
    """
    postsテーブルから全てのoriginal_textを取得する
    """
    cursor = conn.cursor()
    cursor.execute("SELECT original_text FROM posts ORDER BY post_datetime ASC")
    # データベースにテキストが空のレコードがある可能性を考慮
    return [row['original_text'] for row in cursor.fetchall() if row['original_text']]

# --- テキスト処理関連 ---

def normalize_text(text: str) -> str:
    """
    意味内容分析のために、テキストを正規化する
    """
    if not text: return ""
    text = unicodedata.normalize('NFKC', text) # 全角英数字を半角に、カタカナを全角に
    text = text.lower() # 小文字に変換
    text = re.sub(r'[、。！？]', '', text) # 一般的な句読点を除去
    text = re.sub(r'[^\w\s]', ' ', text) # その他の記号をスペースに置換
    text = re.sub(r'\s+', ' ', text).strip() # 連続する空白を1つに
    return text

def tokenize_with_mecab(text: str) -> str:
    """
    MeCabを使ってテキストをわかち書きする (TfidfVectorizerに直接渡せるようにスペース区切りの文字列を返す)
    """
    # mecabrcのパスを明示的に指定して初期化エラーを防ぐ
    tagger = MeCab.Tagger("-Owakati -r /etc/mecabrc")
    return tagger.parse(text).strip()


def prepare_vectorizer(conn: sqlite3.Connection) -> Optional[TfidfVectorizer]:
    """
    TF-IDFベクトル化器を準備する。ファイルが存在すれば読み込み、なければ生成・保存する。
    """
    try:
        # 既存のベクトル化器を読み込み
        with open(config.VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        logging.info(f"ベクトル化器を '{config.VECTORIZER_PATH}' から読み込みました。")
        return vectorizer
    except FileNotFoundError:
        logging.info("ベクトル化器ファイルが見つかりません。新規に生成します。")

        # データベースからコーパスを取得
        corpus = get_all_original_texts(conn)
        if not corpus:
            logging.error("コーパスの取得に失敗しました。postsテーブルにデータが存在するか確認してください。")
            return None

        logging.info(f"{len(corpus)}件の投稿をコーパスとしてベクトル化器を学習させます。")

        # 正規化処理
        normalized_corpus = [normalize_text(text) for text in corpus]

        # TfidfVectorizerの初期化と学習
        # tokenizerにMeCabの関数を指定し、ストップワードも定義
        stop_words = ['です', 'ます', 'の', 'は', 'が', 'を', 'に', 'と', 'も', 'で', 'いる', 'する', 'ある']
        vectorizer = TfidfVectorizer(tokenizer=tokenize_with_mecab, stop_words=stop_words)

        vectorizer.fit(normalized_corpus)

        # 学習済みベクトル化器を保存
        with open(config.VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        logging.info(f"学習済みのベクトル化器を '{config.VECTORIZER_PATH}' に保存しました。")

        return vectorizer
    except Exception as e:
        logging.error(f"ベクトル化器の準備中に予期せぬエラーが発生しました: {e}")
        return None

# --- 評価ロジック Step 1: 高速フィルタリング ---

def get_candidate_posts(conn: sqlite3.Connection, target_post: Post) -> List[Post]:
    """
    高速フィルタリング: ターゲット投稿と静的プロファイルの一部が一致する候補投稿をDBから取得
    """
    cursor = conn.cursor()
    try:
        query = """
            SELECT * FROM posts
            WHERE author_real_gender = ?
              AND author_real_age = ?
              AND author_char_race = ?
              AND author_char_gender = ?
              AND post_id != ?
              AND post_datetime < ?
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
        # Postオブジェクトの生成時にNoneチェックが強化されたため、ここでの追加チェックは不要
        candidate_posts = [Post(dict(row)) for row in rows]
        logging.info(f"高速フィルタリングにより {len(candidate_posts)} 件の比較候補を検出しました。")
        return candidate_posts
    except sqlite3.Error as e:
        logging.error(f"候補投稿の取得中にデータベースエラーが発生しました: {e}")
        return []

# --- 評価ロジック Step 5: 結果の保存と出力 ---

def save_evaluation_score(conn: sqlite3.Connection, post_id: int, result: Dict[str, Any]):
    """
    評価結果を evaluation_scores テーブルに保存する
    """
    cursor = conn.cursor()
    try:
        query = """
            INSERT OR REPLACE INTO evaluation_scores (
                post_id, evaluation_datetime, unique_score, most_similar_post_id,
                max_similarity_score, is_repost, penalty,
                score_breakdown, author_post_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        # score_breakdownはJSON文字列に変換
        score_breakdown_json = json.dumps(result.get('score_breakdown', {}), ensure_ascii=False)

        params = (
            post_id,
            datetime.datetime.now().isoformat(),
            result.get('unique_score'),
            result.get('most_similar_post_id'),
            result.get('max_similarity_score'),
            result.get('is_repost'),
            result.get('penalty'),
            score_breakdown_json,
            result.get('author_post_count')
        )
        cursor.execute(query, params)
        conn.commit()
        logging.info(f"Post ID {post_id} の評価結果をデータベースに保存しました。")
    except sqlite3.Error as e:
        logging.error(f"評価結果の保存中にデータベースエラーが発生しました: {e}")

def print_summary(target_post: Post, result: Dict[str, Any]):
    """
    評価結果のサマリーをコンソールに出力する
    """
    print("\n" + "="*60)
    print(" " * 20 + "評価結果サマリー")
    print("="*60)
    print(f" 評価対象投稿ID: {target_post.post_id}")
    print(f" 投稿者名:         {target_post.author_name}")
    print(f" 投稿日時:         {target_post.post_datetime.strftime('%Y-%m-%d %H:%M') if target_post.post_datetime else 'N/A'}")
    print("-"*60)
    print(f" 【最終ユニークスコア】: {result['unique_score']:.2f} / 100")
    print("-"*60)
    print(f"   - 最大類似度スコア:     {result['max_similarity_score']:.2f}")
    if result['most_similar_post_id']:
        print(f"   - 最も類似した投稿ID: {result['most_similar_post_id']}")
    else:
        print("   - 類似した投稿は見つかりませんでした。")
    print(f"   - 再投稿判定:           {'はい' if result['is_repost'] else 'いいえ'}")
    print(f"   - 適用ペナルティ:       {result['penalty']:.2f}")
    print("="*60 + "\n")


# --- 評価ロジック Step 2: 類似度分析 ---

def calculate_static_profile_score(post_a: Post, post_b: Post) -> float:
    """
    A) 静的プロファイル類似度スコアを計算 (45点満点)
    """
    score = 0
    # リアルプロファイル
    age_diff = abs(post_a.author_real_age - post_b.author_real_age)
    score += config.REAL_AGE_SCORE_MAX * (1 - min(5, age_diff) / 5)

    # キャラ情報
    if post_a.author_char_race == post_b.author_char_race and \
       post_a.author_char_gender == post_b.author_char_gender:
        score += config.CHAR_INFO_SCORE_MAX

    # 変動プロファイル
    if post_a.author_char_job == post_b.author_char_job and \
       post_a.author_name == post_b.author_name:
        score += config.VARIABLE_PROFILE_SCORE_MAX

    return score

def calculate_behavioral_pattern_score(post_a: Post, post_b: Post, tag_weights: Dict[str, float]) -> float:
    """
    B) 行動・嗜好パターン類似度スコアを計算 (25点満点) - 重み付きJaccard係数
    """
    intersection_tags = post_a.all_tags.intersection(post_b.all_tags)
    union_tags = post_a.all_tags.union(post_b.all_tags)

    if not union_tags:
        return 0.0

    intersection_weight = sum(tag_weights.get(tag, 0) for tag in intersection_tags)
    union_weight = sum(tag_weights.get(tag, 0) for tag in union_tags)

    if union_weight == 0:
        return 0.0

    weighted_jaccard_index = intersection_weight / union_weight
    return config.BEHAVIORAL_PATTERN_SCORE_MAX * weighted_jaccard_index

def _calculate_tag_weights_from_db(conn: sqlite3.Connection) -> Dict[str, float]:
    """
    DBから全タグの出現回数を元に、タグの重みを計算する（ヘルパー関数）
    """
    cursor = conn.cursor()
    tag_counts = {}

    # postsテーブルの全レコードを取得
    cursor.execute("SELECT * FROM posts")
    rows = cursor.fetchall()
    all_posts = [Post(dict(row)) for row in rows]

    # 各投稿の全タグ集合を走査してカウント
    for post in all_posts:
        for tag in post.all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # 重みを計算: w(t) = 1 / log(1 + (出現回数))
    tag_weights = {tag: 1 / np.log(1 + count) for tag, count in tag_counts.items()}
    logging.info(f"{len(tag_weights)}個のタグの重みを計算しました。")
    return tag_weights

def prepare_tag_weights(conn: sqlite3.Connection) -> Dict[str, float]:
    """
    タグの重みデータを準備する。ファイルが存在すれば読み込み、なければ生成・保存する。
    """
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
        logging.error(f"タグ重みデータの準備中に予期せぬエラーが発生しました: {e}")
        return {}

def create_stylistic_feature_vector(text: str) -> np.ndarray:
    """
    文体特徴ベクトルを生成する
    """
    if not text:
        return np.zeros(8) # 8次元のゼロベクトル

    # 1. 全角/半角使用比率
    zen_count = len(re.findall(r'[！-～]', text))
    han_count = len(re.findall(r'[!-~]', text))
    zen_han_ratio = zen_count / (zen_count + han_count) if (zen_count + han_count) > 0 else 0

    # 2. 特殊記号・絵文字使用頻度
    special_chars = len(re.findall(r'[！？w♪✨🍀🎀🐰🧸💌]', text))
    special_char_freq = special_chars / len(text)

    # 3. 句読点後スペース率
    punctuations = re.findall(r'[。、]', text)
    punc_with_space = len(re.findall(r'([。、])\s', text))
    punc_space_ratio = punc_with_space / len(punctuations) if punctuations else 0

    # 4. 改行頻度
    newline_freq = text.count('\n') / len(text)

    # 5-7. 各文字種の比率
    hiragana_ratio = len(re.findall(r'[\u3040-\u309F]', text)) / len(text)
    katakana_ratio = len(re.findall(r'[\u30A0-\u30FF]', text)) / len(text)
    kanji_ratio = len(re.findall(r'[\u4E00-\u9FFF]', text)) / len(text)

    # 8. 難読漢字使用率 (簡易版: ここでは固定値0とする)
    # 本来は常用漢字リストとの比較が必要だが、今回は省略
    difficult_kanji_ratio = 0.0

    return np.array([
        zen_han_ratio, special_char_freq, punc_space_ratio, newline_freq,
        hiragana_ratio, katakana_ratio, kanji_ratio, difficult_kanji_ratio
    ])

def calculate_linguistic_fingerprint_score(
    post_a: Post, post_b: Post, vectorizer: TfidfVectorizer
) -> Dict[str, float]:
    """
    C) 言語的指紋類似度スコアと、その内訳を計算する
    """
    # 1. 意味内容の類似度
    texts = [normalize_text(post_a.original_text), normalize_text(post_b.original_text)]
    tfidf_vectors = vectorizer.transform(texts)
    semantic_similarity = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors[1:2])[0][0]
    semantic_score = config.SEMANTIC_SCORE_MAX * semantic_similarity

    # 2. 文体指紋の類似度
    style_vec_a = create_stylistic_feature_vector(post_a.original_text)
    style_vec_b = create_stylistic_feature_vector(post_b.original_text)

    euclidean_dist = np.linalg.norm(style_vec_a - style_vec_b)
    stylistic_similarity = np.exp(-0.5 * euclidean_dist)
    stylistic_score = config.STYLISTIC_SCORE_MAX * stylistic_similarity

    # 3. 信頼度係数の適用
    reliability_factor = min(1.0, len(post_a.original_text) / config.RELIABILITY_TEXT_LENGTH if post_a.original_text else 0.0)

    final_linguistic_score = (semantic_score + stylistic_score) * reliability_factor

    return {
        "total": final_linguistic_score,
        "semantic": semantic_score,
        "stylistic": stylistic_score
    }

# --- 評価ロジック Step 3 & 4: 補正、判定、最終スコア算出 ---

def trace_repost_chain(conn: sqlite3.Connection, start_post_id: int, visited: Set[int]) -> int:
    """
    再投稿の連鎖を再帰的に遡り、クラスタのサイズ（＝投稿回数）を返す。
    """
    if start_post_id is None or start_post_id in visited:
        return 0

    visited.add(start_post_id)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT most_similar_post_id, is_repost
        FROM evaluation_scores
        WHERE post_id = ?
    """, (start_post_id,))
    result = cursor.fetchone()

    # 評価レコードがない、または再投稿でない場合は、この投稿がチェーンの起点
    if not result or not result['is_repost']:
        return 1

    # 再帰的に次の類似投稿を探索
    return 1 + trace_repost_chain(conn, result['most_similar_post_id'], visited)

def evaluate_post(target_post: Post, candidates: List[Post], vectorizer: TfidfVectorizer, tag_weights: Dict[str, float], conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    1件の投稿を評価し、評価結果を辞書で返す
    """
    score_breakdown = {}

    if not candidates:
        logging.info("比較対象が見つからなかったため、ユニークスコアは100点とします。")
        return {
            "unique_score": 100.0, "most_similar_post_id": None, "max_similarity_score": 0.0,
            "is_repost": 0, "penalty": 0.0, "author_post_count": 1,
            "score_breakdown": {"reason": "No candidates"}
        }

    max_similarity_score = 0.0
    most_similar_post_id = None
    best_candidate_scores = {}

    for candidate_post in candidates:
        static_score = calculate_static_profile_score(target_post, candidate_post)
        behavioral_score = calculate_behavioral_pattern_score(target_post, candidate_post, tag_weights)
        linguistic_scores = calculate_linguistic_fingerprint_score(target_post, candidate_post, vectorizer)
        base_similarity_score = static_score + behavioral_score + linguistic_scores["total"]

        consistency_bonus, inconsistency_penalty = 0, 0
        is_server_fake = (re.search(r'(サーバー|鯖).*?(フェイク|偽装|ダミー)', target_post.original_text or "", re.I) or
                          re.search(r'(サーバー|鯖).*?(フェイク|偽装|ダミー)', candidate_post.original_text or "", re.I))

        if not is_server_fake and target_post.server == candidate_post.server:
            consistency_bonus = config.CONSISTENCY_BONUS
        current_similarity_score = base_similarity_score + consistency_bonus

        if current_similarity_score >= config.REPOST_THRESHOLD:
            if linguistic_scores["semantic"] >= 13.0 and linguistic_scores["stylistic"] <= 7.0:
                inconsistency_penalty = config.INCONSISTENCY_PENALTY
                current_similarity_score += inconsistency_penalty

        if current_similarity_score > max_similarity_score:
            max_similarity_score = current_similarity_score
            most_similar_post_id = candidate_post.post_id
            best_candidate_scores = {
                "static_score": static_score, "behavioral_score": behavioral_score,
                "linguistic_score_total": linguistic_scores["total"], "linguistic_semantic": linguistic_scores["semantic"],
                "linguistic_stylistic": linguistic_scores["stylistic"], "consistency_bonus": consistency_bonus,
                "inconsistency_penalty": inconsistency_penalty, "base_similarity": base_similarity_score,
            }

    score_breakdown.update(best_candidate_scores)

    static_score_of_most_similar = best_candidate_scores.get("static_score", 0)
    is_repost = 1 if max_similarity_score >= config.REPOST_THRESHOLD and \
                     static_score_of_most_similar >= config.REPOST_STATIC_SCORE_THRESHOLD else 0

    author_post_count = 1
    if is_repost and most_similar_post_id is not None:
        # 再帰的に再投稿の連鎖を辿り、投稿回数を決定する
        # ここで most_similar_post_id を起点とすることで、過去の投稿クラスタのサイズを取得できる
        # 今回の投稿自身もクラスタの一部なので、+1する
        past_cluster_size = trace_repost_chain(conn, most_similar_post_id, visited=set())
        author_post_count = past_cluster_size + 1
        logging.info(f"類似投稿(ID: {most_similar_post_id})の再投稿チェーンを遡り、過去の投稿回数を {past_cluster_size} と断定。今回は {author_post_count} 回目の投稿として扱います。")
    else:
        # 再投稿ではない場合、これが新しい投稿クラスタの起点となる
        logging.info("再投稿ではないため、投稿回数は 1 とします。")

    penalty = 0.0
    if is_repost and target_post.post_datetime:
        N = author_post_count
        D = (datetime.datetime.now() - target_post.post_datetime).days
        base_penalty = config.BASE_PENALTY_COEFFICIENT - (config.PENALTY_PER_REPOST * (N - 1))
        recovery_limit = max(config.MIN_RECOVERY_LIMIT, config.MAX_RECOVERY_LIMIT - (N - 2))
        time_recovery_bonus = min(recovery_limit, (D / config.DAYS_FOR_FULL_RECOVERY) * recovery_limit)
        penalty = base_penalty + time_recovery_bonus

    score_breakdown.update({
        "final_penalty": penalty,
        "final_author_post_count": author_post_count
    })

    tentative_score = 100 - max_similarity_score + penalty
    unique_score = max(0, min(100, tentative_score))

    return {
        "unique_score": unique_score, "most_similar_post_id": most_similar_post_id,
        "max_similarity_score": max_similarity_score, "is_repost": is_repost,
        "penalty": penalty, "author_post_count": author_post_count,
        "score_breakdown": score_breakdown
    }

def setup_database_tables(conn: sqlite3.Connection):
    """
    必要なテーブルとインデックスを作成・更新する。
    """
    cursor = conn.cursor()

    # 評価結果を保存するテーブル (基本構造)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluation_scores (
        post_id INTEGER PRIMARY KEY,
        evaluation_datetime TEXT,
        unique_score REAL,
        most_similar_post_id INTEGER,
        max_similarity_score REAL,
        is_repost INTEGER,
        penalty REAL
    );
    """)

    # --- カラムの追加 (冪等性を担保) ---
    # 現在のテーブル情報を取得
    cursor.execute("PRAGMA table_info(evaluation_scores);")
    existing_columns = [row['name'] for row in cursor.fetchall()]

    # score_breakdown カラムの追加
    if 'score_breakdown' not in existing_columns:
        cursor.execute("ALTER TABLE evaluation_scores ADD COLUMN score_breakdown TEXT;")
        logging.info("'score_breakdown' カラムを evaluation_scores テーブルに追加しました。")

    # author_post_count カラムの追加
    if 'author_post_count' not in existing_columns:
        cursor.execute("ALTER TABLE evaluation_scores ADD COLUMN author_post_count INTEGER;")
        logging.info("'author_post_count' カラムを evaluation_scores テーブルに追加しました。")

    # Step 1 高速フィルタリング用のインデックス
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_post_filter
    ON posts (author_real_gender, author_real_age, author_char_race, author_char_gender);
    """)

    conn.commit()
    logging.info("データベーステーブルのセットアップが完了しました。")

# --- メイン処理 ---
def main():
    """
    評価プログラムのメイン処理
    """
    parser = argparse.ArgumentParser(description="投稿のユニークスコアを評価するプログラム")
    parser.add_argument('--post-id', type=int, help='指定したpost_idの投稿のみを評価する')
    parser.add_argument('--re-evaluate-all', action='store_true', help='すべての投稿を強制的に再評価する')
    args = parser.parse_args()

    logging.info("--- 投稿評価プログラムを開始します ---")

    try:
        with sqlite3.connect(config.DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            setup_database_tables(conn)

            # Step 1: 評価対象の選定 (argparse対応)
            target_posts = get_posts_to_evaluate(conn, args)
            if not target_posts:
                logging.info("評価対象が見つからなかったため、処理を終了します。")
                return

            # Step 2: TF-IDFコーパスとタグ重みの準備
            vectorizer = prepare_vectorizer(conn)
            if not vectorizer:
                logging.error("ベクトル化器の準備に失敗したため、処理を中断します。")
                return
            tag_weights = prepare_tag_weights(conn)


            # Step 3: 評価ループ
            for i, target_post in enumerate(target_posts):
                logging.info(f"--- ({i+1}/{len(target_posts)}) post_id: {target_post.post_id} の評価を開始 ---")

                # Step 3-1 (高速フィルタリング)
                candidate_posts = get_candidate_posts(conn, target_post)

                # Step 3-2 (類似度分析〜最終スコア算出)
                evaluation_result = evaluate_post(target_post, candidate_posts, vectorizer, tag_weights, conn)

                # Step 3-3 (結果の保存と出力)
                save_evaluation_score(conn, target_post.post_id, evaluation_result)
                print_summary(target_post, evaluation_result)

    except sqlite3.Error as e:
        logging.error(f"データベース処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"プログラムの実行中に予期せぬエラーが発生しました: {e}", exc_info=True)
    finally:
        logging.info("--- 投稿評価プログラムを終了します ---")

if __name__ == "__main__":
    main()
