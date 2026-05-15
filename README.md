# 販売予測モデル ダッシュボード（デモ）

Streamlit 製のデモダッシュボード。ダミーデータのみを使用。
**パスワード認証付き**（合言葉を知っている人だけ閲覧可能）。

## ローカル起動

```bash
pip install -r requirements.txt

# 認証パスワードの設定（一度だけ）
cp app/.streamlit/secrets.toml.example app/.streamlit/secrets.toml
# secrets.toml を開いて password = "好きな合言葉" に書き換える

streamlit run app/app.py
```

## GitHub への push

1. https://github.com/new で新規リポジトリを作成
   - Repository name: `sales-forecast-dashboard-demo`（任意）
   - Public を選択
   - README / .gitignore / license は **追加しない**
2. ターミナルで実行：

   ```bash
   cd /Users/yusuke/Documents/Claude/code-001/dashboard-deploy
   git remote add origin https://github.com/<USER>/sales-forecast-dashboard-demo.git
   git push -u origin main
   ```

## Streamlit Community Cloud デプロイ

1. https://share.streamlit.io にサインイン（GitHub アカウント）
2. "Create app" → "Deploy a public app from GitHub"
3. Repository を選択、Branch=`main`、Main file path=`app/app.py`
4. **重要**: Deploy ボタンを押す前に **Advanced settings → Secrets** を開き、以下を貼り付け：

   ```toml
   password = "ベンダーに渡す合言葉"
   ```

5. Deploy → 数分後に `https://<name>.streamlit.app` が発行される

これでアプリは Public（URL を知れば誰でもアクセス可）になりますが、**パスワードを知らないとダッシュボード本体は見られません**。

## ベンダー共有

1. 発行された URL: `https://<name>.streamlit.app`
2. 設定した合言葉

をメール等で別途送付。ベンダー側は URL を開き、パスワード入力欄に合言葉を入れるだけで閲覧可。

## パスワード変更

Streamlit Cloud 管理画面 → 該当アプリの "Settings" → "Secrets" を編集して保存するだけ。アプリは自動で再起動し、新しいパスワードが有効になる。
