# -*- coding: utf-8 -*-
import sqlite3
import datetime
import traceback
import logging

# --- ロギング設定 ---
# このモジュールでは、呼び出し元のロガーを主に使用する想定
db_logger = logging.getLogger(__name__)

def setup_database(db_name="fap_posts.db"):
    """
    データベースとテーブル（posts, tags, post_tags, system_logs）をセットアップする。
    冪等性を持ち、何度実行しても安全なように設計されている。
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 1. postsテーブル (JSONカラムを削除)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        post_id INTEGER PRIMARY KEY,
        post_datetime TEXT,
        title TEXT,
        purpose TEXT,
        original_text TEXT,
        author_name TEXT,
        author_real_age TEXT,
        author_real_gender TEXT,
        author_char_race TEXT,
        author_char_gender TEXT,
        author_char_job TEXT,
        server TEXT,
        voice_chat TEXT,
        server_transfer TEXT,
        sub_char_ok INTEGER
    )
    """)

    # 2. tagsマスタテーブル (タグを一元管理)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tags (
        tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tag_category TEXT NOT NULL,
        tag_name TEXT NOT NULL,
        UNIQUE(tag_category, tag_name)
    )
    """)

    # 3. post_tags関連テーブル (投稿とタグを紐付け)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS post_tags (
        post_id INTEGER NOT NULL,
        tag_id INTEGER NOT NULL,
        PRIMARY KEY (post_id, tag_id),
        FOREIGN KEY (post_id) REFERENCES posts (post_id) ON DELETE CASCADE,
        FOREIGN KEY (tag_id) REFERENCES tags (tag_id) ON DELETE CASCADE
    )
    """)
    # 検索パフォーマンス向上のためのインデックス
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_post_tags_tag_id ON post_tags (tag_id);")


    # 4. system_logsテーブル (エラーログ記録用)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        level TEXT NOT NULL,
        source TEXT NOT NULL,
        message TEXT NOT NULL,
        details TEXT
    )
    """)

    conn.commit()
    return conn

def log_to_db(db_conn: sqlite3.Connection, level: str, source: str, message: str, details: str = None):
    """
    システムログをデータベースのsystem_logsテーブルに書き込む。
    """
    cursor = db_conn.cursor()
    try:
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO system_logs (timestamp, level, source, message, details)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, level, source, message, details))
        db_conn.commit()
    except sqlite3.Error as e:
        # DBへのロギング自体が失敗した場合は、標準のロギング機能にフォールバック
        db_logger.error(f"Failed to write log to database: {e}")
        db_logger.error(f"Original log message: [{level}] {source} - {message}")

# --- mainブロック（テスト用） ---
if __name__ == '__main__':
    print("データベースユーティリティモジュールのテストを実行します。")
    DB_NAME = "test_fap_posts.db"
    try:
        # 1. データベース設定のテスト
        print(f"'{DB_NAME}' をセットアップ中...")
        conn = setup_database(DB_NAME)
        print("セットアップ完了。")

        # 2. ログ記録機能のテスト
        print("ログ記録機能をテスト中...")
        log_to_db(conn, 'INFO', 'db_utils_test', 'This is a test log message.')
        log_to_db(conn, 'ERROR', 'db_utils_test', 'This is a test error message.', traceback.format_exc())
        print("ログ記録完了。")

        # 3. 結果の確認
        print("system_logsテーブルの内容を確認:")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM system_logs")
        for row in cursor.fetchall():
            print(row)

        conn.close()
        print("\nテスト成功。")

    except Exception as e:
        print(f"\nテスト中にエラーが発生しました: {e}")
    finally:
        # テスト用データベースファイルをクリーンアップ
        import os
        if os.path.exists(DB_NAME):
            os.remove(DB_NAME)
            print(f"テストデータベース '{DB_NAME}' を削除しました。")
