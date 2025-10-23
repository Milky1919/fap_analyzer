# -*- coding: utf-8 -*-

# --- データベース・ファイルパス設定 ---
DB_NAME = "fap_posts.db"
VECTORIZER_PATH = "vectorizer.pkl"
TAG_WEIGHTS_PATH = "tag_weights.pkl"

# --- 類似度スコア計算の重み ---
# 各スコアの満点 (合計が100になるように調整)
# A) 静的プロファイル類似度スコア
STATIC_PROFILE_SCORE_MAX = 45
#   - リアル年代 (満点: 35)
REAL_AGE_SCORE_MAX = 35
#   - キャラ情報 (満点: 5)
CHAR_INFO_SCORE_MAX = 5
#   - 変動プロファイル (満点: 5)
VARIABLE_PROFILE_SCORE_MAX = 5

# B) 行動・嗜好パターン類似度スコア (満点: 25)
BEHAVIORAL_PATTERN_SCORE_MAX = 25

# C) 言語的指紋類似度スコア (合計: 30)
LINGUISTIC_FINGERPRINT_SCORE_MAX = 30
#   - 意味内容の類似度 (満点: 15)
SEMANTIC_SCORE_MAX = 15
#   - 文体指紋の類似度 (満点: 15)
STYLISTIC_SCORE_MAX = 15


# --- 類似度スコアの補正値 ---
# 一貫性ボーナス (サーバー情報が一致した場合)
CONSISTENCY_BONUS = 5
# 不一致ボーナス (新規ユーザー救済措置)
INCONSISTENCY_PENALTY = -5


# --- 再投稿判定のしきい値 ---
# この点数以上で「高確率で再投稿」と判定
REPOST_THRESHOLD = 88.0
# 再投稿判定の条件に含める、静的プロファイルスコアの最低値
REPOST_STATIC_SCORE_THRESHOLD = 35.0


# --- ペナルティ計算の係数 ---
# N: 今回を含めた実質的な再投稿回数
# D: 最新の投稿日時からの経過日数

# 基本ペナルティ = BASE_PENALTY_COEFFICIENT - (PENALTY_PER_REPOST * (N - 1))
BASE_PENALTY_COEFFICIENT = -30
PENALTY_PER_REPOST = 15

# 時間経過によるペナルティ回復
# 回復上限 = max(MIN_RECOVERY_LIMIT, MAX_RECOVERY_LIMIT - (N - 2))
MIN_RECOVERY_LIMIT = 5
MAX_RECOVERY_LIMIT = 15
# 回復量 = min(回復上限, (D / DAYS_FOR_FULL_RECOVERY) * 回復上限)
DAYS_FOR_FULL_RECOVERY = 365.0
# ペナルティの時間回復が始まるまでの日数（この期間中は回復量ゼロ）
NO_RECOVERY_PERIOD_DAYS = 90
# 再投稿とみなさない猶予期間（分）
REPOST_GRACE_PERIOD_MINUTES = 20


# --- その他 ---
# 言語的指紋スコアの信頼度係数を計算する際の基準文字数
RELIABILITY_TEXT_LENGTH = 100
