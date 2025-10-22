import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re
import json
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_URL = "https://find-fr.com/"
DB_NAME = "fap_posts.db"

def setup_database():
    """データベースとテーブルをセットアップする"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        post_id INTEGER PRIMARY KEY, post_datetime TEXT, title TEXT, purpose TEXT, original_text TEXT,
        author_name TEXT, author_real_age TEXT, author_real_gender TEXT, author_char_race TEXT,
        author_char_gender TEXT, author_char_job TEXT, server TEXT, voice_chat TEXT,
        external_tools TEXT, playstyle_tags TEXT, server_transfer TEXT, sub_char_ok INTEGER,
        activity_times TEXT, wish_races TEXT, wish_char_genders TEXT, wish_jobs TEXT,
        wish_real_genders TEXT, wish_real_ages TEXT
    )
    """)
    conn.commit()
    return conn

def get_text_or_none(element, selector):
    """セレクタに一致する要素のテキストを返す。なければNone"""
    if not element:
        return None
    tag = element.select_one(selector)
    return tag.text.strip() if tag else None

def get_all_texts_as_json(element, selector):
    """セレクタに一致する全要素のテキストをリスト化し、JSON文字列で返す"""
    if not element:
        return '[]'
    tags = element.select(selector)
    return json.dumps([tag.text.strip() for tag in tags], ensure_ascii=False) if tags else '[]'

def find_dd_by_dt_text(element, dt_text):
    """
    指定されたテキストを持つdt要素を検索し、その次にあるdd要素を返す。
    :param element: 検索対象のBeautifulSoup要素
    :param dt_text: 検索するdt要素のテキスト
    :return: 見つかったdd要素、またはNone
    """
    if not element:
        return None

    # dtをすべて探す
    dt_tags = element.find_all("dt")
    for dt in dt_tags:
        # テキストをstripして比較
        if dt.get_text(strip=True) == dt_text:
            # dd要素はdtの直接の兄弟要素であると仮定
            dd = dt.find_next_sibling('dd')
            return dd
    return None

def parse_post(post_section):
    """1つの投稿セクションから全てのデータを抽出し、辞書として返す"""
    data = {}
    try:
        # --- 基本情報 ---
        post_id_tag = post_section.select_one("h2 a[href^='/post/detail/']")
        if not post_id_tag: return None
        match = re.search(r'/(\d+)$', post_id_tag['href'])
        # 投稿IDがなければ、この投稿はスキップ
        if not match: return None
        data['post_id'] = int(match.group(1))

        data['post_datetime'] = get_text_or_none(post_section, "span.list_created_time > time")

        title_h2 = post_section.select_one("h2[class*='purpose']")
        if title_h2:
            # h2直下のテキストノード（spanタグ内を含まない）を取得
            title_text_node = title_h2.find(string=True, recursive=False)
            data['title'] = title_text_node.strip() if title_text_node else ""
        else:
            data['title'] = None

        data['purpose'] = get_text_or_none(post_section, ".tag_sv > p[class*='purpose']")
        data['original_text'] = get_text_or_none(post_section, ".pr_comment p")

        # --- 募集主プロフィール ---
        profile_p = post_section.select_one(".profile p")
        if profile_p:
            span_tag = profile_p.select_one("span")
            if span_tag:
                # 以前の実装ではNavigableStringオブジェクトでない場合にエラーになる可能性があった
                prev_sibling = span_tag.previous_sibling
                data['author_name'] = prev_sibling.strip() if prev_sibling and isinstance(prev_sibling, str) else ''

                real_profile_text = span_tag.text
                age_match = re.search(r'(\S+代)', real_profile_text)
                gender_match = re.search(r'(男性|女性)', real_profile_text)
                data['author_real_age'] = age_match.group(1) if age_match else '？代'
                data['author_real_gender'] = gender_match.group(1) if gender_match else '性別非公開'
            else:
                data['author_name'] = None
                data['author_real_age'] = '？代'
                data['author_real_gender'] = '性別非公開'

            br_tag = profile_p.find('br')
            if br_tag and br_tag.next_sibling and isinstance(br_tag.next_sibling, str):
                full_race_text = br_tag.next_sibling.strip().split('/')[0].strip()
                data['author_char_race'] = full_race_text
            else:
                data['author_char_race'] = None

            data['author_char_gender'] = get_text_or_none(profile_p, "b")

            full_profile_text = profile_p.get_text(separator=' ', strip=True)
            job_match = re.search(r'/\s*(.+)$', full_profile_text)
            data['author_char_job'] = job_match.group(1).strip() if job_match else None

        # --- プレイ環境・スタイル (堅牢性向上) ---
        server_tags = post_section.select(".tag_sv > p.server")

        server_name_tag = next((tag for tag in server_tags if "移転" not in tag.text.strip()), None)
        data['server'] = server_name_tag.text.strip() if server_name_tag else None

        transfer_tag = next((tag for tag in server_tags if "移転" in tag.text.strip()), None)
        data['server_transfer'] = transfer_tag.text.strip() if transfer_tag else '移転不可'

        data['sub_char_ok'] = 1 if post_section.select_one(".tag_sv > p.sub") else 0

        # --- :contains() を使わない堅牢な抽出 ---
        vc_dd = find_dd_by_dt_text(post_section, 'ボイスチャット')
        data['voice_chat'] = vc_dd.get_text(strip=True) if vc_dd else None

        et_dd = find_dd_by_dt_text(post_section, '外部ツール')
        data['external_tools'] = get_all_texts_as_json(et_dd, ".choices_list")

        data['playstyle_tags'] = get_all_texts_as_json(post_section, ".tag_pr > p")

        # --- 希望条件 ---
        at_dd = find_dd_by_dt_text(post_section, '活動時間')
        data['activity_times'] = get_all_texts_as_json(at_dd, ".choices_list")

        wr_dd = find_dd_by_dt_text(post_section, '希望種族')
        data['wish_races'] = get_all_texts_as_json(wr_dd, ".choices_list")

        wcg_dd = find_dd_by_dt_text(post_section, '希望キャラ性別')
        data['wish_char_genders'] = get_all_texts_as_json(wcg_dd, ".choices_list")

        wj_dd = find_dd_by_dt_text(post_section, '希望ジョブ')
        data['wish_jobs'] = get_all_texts_as_json(wj_dd, ".choices_list")

        wrg_dd = find_dd_by_dt_text(post_section, '希望リアル性別')
        data['wish_real_genders'] = get_all_texts_as_json(wrg_dd, ".choices_list")

        wra_dd = find_dd_by_dt_text(post_section, '希望リアル年代')
        data['wish_real_ages'] = get_all_texts_as_json(wra_dd, ".choices_list")

        return data
    except Exception as e:
        logging.error(f"投稿ID {data.get('post_id', 'N/A')} の解析中にエラー: {e}", exc_info=True)
        return None

def save_to_db(data, conn):
    """抽出したデータをデータベースに保存する"""
    cursor = conn.cursor()
    valid_data = {k: v for k, v in data.items() if v is not None}
    columns = ', '.join(valid_data.keys())
    placeholders = ', '.join('?' * len(valid_data))
    sql = f"INSERT OR REPLACE INTO posts ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, list(valid_data.values()))
    conn.commit()

def main(start_page=1, end_page=1):
    """メインのスクレイピング処理"""
    logging.info("スクレイピング処理を開始します。")
    conn = setup_database()

    for page_num in range(start_page, end_page + 1):
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

            for post_section in posts:
                post_data = parse_post(post_section)
                if post_data and post_data.get('post_id'):
                    save_to_db(post_data, conn)

            logging.info(f"{page_num}ページ目から{len(posts)}件の投稿を処理しました。")
        except requests.RequestException as e:
            logging.error(f"{page_num}ページ目の取得に失敗: {e}")

        time.sleep(5)

    conn.close()
    logging.info("スクレイピング処理が完了しました。")

if __name__ == "__main__":
    # --- 実行と検証の手順 ---
    # 1. まずは1ページだけでテスト実行
    main(start_page=1, end_page=10)

    # 2. DB Browser for SQLiteでfap_posts.dbを開き、データが正しく保存されているか確認

    # 3. 1ページのテストが成功したら、全ページを対象に実行
    # main(start_page=1, end_page=143)
