# 販売予測モデル ダッシュボード（デモ）

Streamlit 製のデモダッシュボード。ダミーデータのみを使用。

## ローカル起動

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## GitHub への push 手順（gh CLI 未インストールのため Web 経由）

1. https://github.com/new で新規リポジトリを作成
   - Repository name: `sales-forecast-dashboard-demo`（任意）
   - Public を選択
   - README / .gitignore / license は **追加しない**（既にローカルにあるため）
2. ターミナルで以下を実行（`<USER>` は GitHub ユーザー名）

   ```bash
   cd /Users/yusuke/Documents/Claude/code-001/dashboard-deploy
   git remote add origin git@github.com:<USER>/sales-forecast-dashboard-demo.git
   # HTTPS の場合: git remote add origin https://github.com/<USER>/sales-forecast-dashboard-demo.git
   git push -u origin main
   ```

## Streamlit Community Cloud デプロイ

1. https://share.streamlit.io でサインイン（GitHub アカウント）
2. "Create app" → "Deploy a public app from GitHub"
3. Repository を選択、Branch=`main`、Main file path=`app/app.py`
4. Deploy → 数分後に `https://<name>.streamlit.app` の URL が発行される
5. App settings → Sharing で閲覧者を自分の Google アカウントに限定可能

## 別 PC からの閲覧

発行された `https://<name>.streamlit.app` を別 PC のブラウザで開く。
共有制限を設定した場合は同じ Google アカウントでサインインが必要。
