# ベースとなるOSとPythonのバージョンを指定
FROM python:3.10-slim

# 環境変数の設定（Pythonのログ出力を調整）
ENV PYTHONUNBUFFERED 1

# システムパッケージの更新と、MeCabおよびビルドに必要なツールのインストール
RUN apt-get update && apt-get install -y \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    git \
    make \
    curl \
    g++ \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# aptでインストールされたmecabrcへのシンボリックリンクを作成し、mecab-python3が参照できるようにする
RUN ln -s /etc/mecabrc /usr/local/etc/mecabrc

# プロジェクトファイルを格納するディレクトリを作成
WORKDIR /app

# 必要なPythonライブラリをインストールするためのリストをコピー
COPY requirements.txt .

# pipを使ってPythonライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# APIが使用するポートを公開
EXPOSE 8000

# コンテナの起動時に実行されるデフォルトのコマンド（何もしないで待機）
CMD ["tail", "-f", "/dev/null"]
