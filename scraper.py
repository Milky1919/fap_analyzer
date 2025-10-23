import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re
import logging
import traceback
import datetime
import json

# 共有モジュールをインポート
import db_utils
import config

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 定数定義 ---
BASE_URL = "https://find-fr.com/"

# --- HTML解析ヘルパー関数 ---

def get_text_or_none(element, selector):
    """セレクタに一致する要素のテキストを返す。なければNone"""
    if not element:
        return None
    tag = element.select_one(selector)
    return tag.text.strip() if tag else None

def get_all_texts_as_list(element, selector):
    """セレクタに一致する全要素のテキストをリストとして返す"""
    if not element:
        return []
    tags = element.select(selector)
    return [tag.text.strip() for tag in tags] if tags else []

def find_dd_by_dt_text(element, dt_text):
    """指定されたテキストを持つdt要素を検索し、その次にあるdd要素を返す"""
    if not element:
        return None
    dt_tags = element.find_all("dt")
    for dt in dt_tags:
        if dt.get_text(strip=True) == dt_text:
            dd = dt.find_next_sibling('dd')
            return dd
    return None

def parse_post(post_section):
    """
    1つの投稿セクションから全てのデータを抽出し、辞書として返す。
    タグ関連データは'tags'キーの中にカテゴリ分けして格納する。
    """
    post_data = {}
    tag_data = {}

    # --- 基本情報 (post_data) ---
    post_id_tag = post_section.select_one("h2 a[href^='/post/detail/']")
    if not post_id_tag: return None
    match = re.search(r'/(\d+)$', post_id_tag['href'])
    if not match: return None
    post_data['post_id'] = int(match.group(1))

    post_data['post_datetime'] = get_text_or_none(post_section, "span.list_created_time > time")
    title_h2 = post_section.select_one("h2[class*='purpose']")
    if title_h2:
        title_text_node = title_h2.find(string=True, recursive=False)
        post_data['title'] = title_text_node.strip() if title_text_node else ""
    else:
        post_data['title'] = None
    post_data['purpose'] = get_text_or_none(post_section, ".tag_sv > p[class*='purpose']")
    post_data['original_text'] = get_text_or_none(post_section, ".pr_comment p")

    # --- 募集主プロフィール (post_data) ---
    profile_p = post_section.select_one(".profile p")
    if profile_p:
        span_tag = profile_p.select_one("span")
        if span_tag:
            prev_sibling = span_tag.previous_sibling
            post_data['author_name'] = prev_sibling.strip() if prev_sibling and isinstance(prev_sibling, str) else ''
            real_profile_text = span_tag.text
            age_match = re.search(r'(\S+代)', real_profile_text)
            gender_match = re.search(r'(男性|女性)', real_profile_text)
            post_data['author_real_age'] = age_match.group(1) if age_match else '？代'
            post_data['author_real_gender'] = gender_match.group(1) if gender_match else '性別非公開'
        else:
            post_data['author_name'] = None
            post_data['author_real_age'] = '？代'
            post_data['author_real_gender'] = '性別非公開'
        br_tag = profile_p.find('br')
        if br_tag and br_tag.next_sibling and isinstance(br_tag.next_sibling, str):
            post_data['author_char_race'] = br_tag.next_sibling.strip().split('/')[0].strip()
        else:
            post_data['author_char_race'] = None
        post_data['author_char_gender'] = get_text_or_none(profile_p, "b")
        full_profile_text = profile_p.get_text(separator=' ', strip=True)
        job_match = re.search(r'/\s*(.+)$', full_profile_text)
        post_data['author_char_job'] = job_match.group(1).strip() if job_match else None

    # --- プレイ環境・スタイル (post_data) ---
    server_tags = post_section.select(".tag_sv > p.server")
    server_name_tag = next((tag for tag in server_tags if "移転" not in tag.text.strip()), None)
    post_data['server'] = server_name_tag.text.strip() if server_name_tag else None
    transfer_tag = next((tag for tag in server_tags if "移転" in tag.text.strip()), None)
    post_data['server_transfer'] = transfer_tag.text.strip() if transfer_tag else '移転不可'
    post_data['sub_char_ok'] = 1 if post_section.select_one(".tag_sv > p.sub") else 0
    vc_dd = find_dd_by_dt_text(post_section, 'ボイスチャット')
    post_data['voice_chat'] = vc_dd.get_text(strip=True) if vc_dd else None

    # --- タグ情報 (tag_data) ---
    et_dd = find_dd_by_dt_text(post_section, '外部ツール')
    tag_data['external_tools'] = get_all_texts_as_list(et_dd, ".choices_list")

    tag_data['playstyle_tags'] = get_all_texts_as_list(post_section, ".tag_pr > p")

    at_dd = find_dd_by_dt_text(post_section, '活動時間')
    tag_data['activity_times'] = get_all_texts_as_list(at_dd, ".choices_list")

    wr_dd = find_dd_by_dt_text(post_section, '希望種族')
    tag_data['wish_races'] = get_all_texts_as_list(wr_dd, ".choices_list")

    wcg_dd = find_dd_by_dt_text(post_section, '希望キャラ性別')
    tag_data['wish_char_genders'] = get_all_texts_as_list(wcg_dd, ".choices_list")

    wj_dd = find_dd_by_dt_text(post_section, '希望ジョブ')
    tag_data['wish_jobs'] = get_all_texts_as_list(wj_dd, ".choices_list")

    wrg_dd = find_dd_by_dt_text(post_section, '希望リアル性別')
    tag_data['wish_real_genders'] = get_all_texts_as_list(wrg_dd, ".choices_list")

    wra_dd = find_dd_by_dt_text(post_section, '希望リアル年代')
    tag_data['wish_real_ages'] = get_all_texts_as_list(wra_dd, ".choices_list")

    # 抽出した2種類のデータを結合して返す
    return {'post_data': post_data, 'tag_data': tag_data}


# --- データベース保存ロジック ---

def get_or_create_tag(cursor, category, name):
    """tagsテーブルにタグが存在すればtag_idを返し、なければ作成してtag_idを返す"""
    cursor.execute("SELECT tag_id FROM tags WHERE tag_category = ? AND tag_name = ?", (category, name))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute("INSERT INTO tags (tag_category, tag_name) VALUES (?, ?)", (category, name))
        return cursor.lastrowid

def save_to_db(data, conn):
    """
    差分更新と履歴保存ロジックを実装したDB保存関数。
    - 新規投稿: postsとpost_tagsにINSERT
    - 既存投稿:
        - 変更があれば、古いデータをpost_historyに保存し、postsとpost_tagsをUPDATE
        - 変更がなければ、何もしない
    @return: 投稿が既存だった場合はTrue、新規だった場合はFalseを返す
    """
    cursor = conn.cursor()
    post_data = data['post_data']
    tag_data = data['tag_data']
    post_id = post_data['post_id']

    # --- 既存データの取得 ---
    cursor.execute("SELECT * FROM posts WHERE post_id = ?", (post_id,))
    existing_post_row = cursor.fetchone()

    # --- 新規投稿の場合 ---
    if not existing_post_row:
        columns = ', '.join(post_data.keys())
        placeholders = ', '.join('?' * len(post_data))
        sql = f"INSERT INTO posts ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, list(post_data.values()))

        # タグの登録
        for category, tags in tag_data.items():
            for tag_name in tags:
                tag_id = get_or_create_tag(cursor, category, tag_name)
                cursor.execute("INSERT OR IGNORE INTO post_tags (post_id, tag_id) VALUES (?, ?)", (post_id, tag_id))
        conn.commit()
        return False # 新規

    # --- 既存投稿の場合 ---
    existing_post = dict(existing_post_row)
    # 既存タグの取得
    cursor.execute("""
        SELECT t.tag_category, t.tag_name
        FROM post_tags pt JOIN tags t ON pt.tag_id = t.tag_id
        WHERE pt.post_id = ?
    """, (post_id,))
    existing_tags = {}
    for row in cursor.fetchall():
        category, name = row[0], row[1]
        if category not in existing_tags: existing_tags[category] = []
        existing_tags[category].append(name)

    # --- 変更点の比較 ---
    is_post_changed = any(str(existing_post.get(k, '')) != str(v) for k, v in post_data.items())

    # タグの比較 (順序を無視して比較するためにsetを使用)
    new_tags_set = {cat: set(tags) for cat, tags in tag_data.items()}
    old_tags_set = {cat: set(tags) for cat, tags in existing_tags.items()}
    is_tags_changed = new_tags_set != old_tags_set

    if not is_post_changed and not is_tags_changed:
        logging.info(f"Post ID {post_id} は変更がないためスキップします。")
        return True # 既存

    # --- 変更がある場合のみ、履歴保存と更新を実行 ---
    logging.info(f"Post ID {post_id} に変更を検出。履歴を保存し、データを更新します。")

    # 1. 古いデータをpost_historyにアーカイブ
    history_data = existing_post
    history_data['archived_at'] = datetime.datetime.now().isoformat()
    history_data['tags_json'] = json.dumps(existing_tags, ensure_ascii=False)
    hist_cols = ', '.join(history_data.keys())
    hist_placeholders = ', '.join('?' * len(history_data))
    cursor.execute(f"INSERT INTO post_history ({hist_cols}) VALUES ({hist_placeholders})", list(history_data.values()))

    # 2. postsテーブルを新しいデータでUPDATE
    set_clause = ', '.join([f"{key} = ?" for key in post_data.keys()])
    cursor.execute(f"UPDATE posts SET {set_clause} WHERE post_id = ?", list(post_data.values()) + [post_id])

    # 3. post_tagsテーブルを更新 (DELETE & INSERT)
    cursor.execute("DELETE FROM post_tags WHERE post_id = ?", (post_id,))
    for category, tags in tag_data.items():
        for tag_name in tags:
            tag_id = get_or_create_tag(cursor, category, tag_name)
            cursor.execute("INSERT OR IGNORE INTO post_tags (post_id, tag_id) VALUES (?, ?)", (post_id, tag_id))

    conn.commit()
    return True # 既存


# --- メイン処理 ---

def main(start_page=1, end_page=100): # end_pageのデフォルト値を増やしておく
    """メインのスクレイピング処理（差分更新対応）"""
    logging.info("スクレイピング処理を開始します。")
    conn = None
    try:
        conn = db_utils.setup_database(config.DB_NAME)
        # Row factoryを設定して辞書ライクなアクセスを可能に
        conn.row_factory = sqlite3.Row
        logging.info("データベースのセットアップが完了しました。")

        stop_scraping = False
        for page_num in range(start_page, end_page + 1):
            if stop_scraping:
                logging.info("既存の投稿を検出したため、次のページの処理をスキップし、スクレイピングを終了します。")
                break

            target_url = f"{BASE_URL}?p={page_num}"
            logging.info(f"{page_num}ページ目を処理中: {target_url}")

            try:
                response = requests.get(target_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')

                posts = soup.select("section.recruit_content")
                if not posts:
                    logging.warning(f"{page_num}ページ目に投稿なし。処理を終了します。")
                    break

                processed_count = 0
                for post_section in posts:
                    try:
                        parsed_data = parse_post(post_section)
                        if parsed_data and parsed_data['post_data'].get('post_id'):
                            is_existing = save_to_db(parsed_data, conn)
                            if is_existing:
                                stop_scraping = True # 既存の投稿が見つかったフラグを立てる
                            processed_count += 1
                    except Exception as e:
                        post_id_str = f"post_id: {parsed_data.get('post_data', {}).get('post_id', 'N/A')}" if 'parsed_data' in locals() else "Unknown post"
                        error_message = f"単一投稿の解析または保存中にエラー: {post_id_str}"
                        logging.error(error_message, exc_info=True)
                        db_utils.log_to_db(conn, 'ERROR', 'scraper.py', error_message, traceback.format_exc())

                logging.info(f"{page_num}ページ目から{processed_count}件の投稿を処理しました。")

            except requests.RequestException as e:
                logging.error(f"{page_num}ページ目の取得に失敗: {e}")
                db_utils.log_to_db(conn, 'ERROR', 'scraper.py', f"HTTPリクエスト失敗: {e}", traceback.format_exc())

            time.sleep(5)

    except Exception as e:
        logging.critical("スクレイピング処理の途中で致命的なエラーが発生しました。", exc_info=True)
        if conn:
            db_utils.log_to_db(conn, 'CRITICAL', 'scraper.py', f"致命的なエラー: {e}", traceback.format_exc())

    finally:
        if conn:
            conn.close()
        logging.info("スクレイピング処理が完了しました。")


if __name__ == "__main__":
    # --- 実行と検証の手順 ---
    # 1. 1〜2ページでテスト実行
    main(start_page=1, end_page=10)

    # 2. DB Browser for SQLiteでfap_posts.dbを開き、以下の点を確認
    #    - postsテーブルからJSON形式のカラムが消えているか
    #    - tagsテーブルにタグがカテゴリと名前で保存されているか
    #    - post_tagsテーブルにposts.post_idとtags.tag_idの対応が記録されているか
    #    - system_logsテーブル (意図的にエラーを発生させない限りは空のはず)

    # 3. テストが成功したら、必要に応じて全ページを対象に実行
    # main(start_page=1, end_page=143)
