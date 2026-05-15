# 販売予測モデル ダッシュボード（デモ）

Streamlit 製のデモダッシュボード。ダミーデータのみを使用。

## ローカル起動

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Streamlit Community Cloud デプロイ

1. このリポジトリを GitHub に push（public）
2. https://share.streamlit.io でサインイン → "New app"
3. Repository を選択、Branch=`main`、Main file path=`app/app.py`
4. Deploy → 数分後に `https://<name>.streamlit.app` の URL が発行される
5. App settings → Sharing で閲覧者を自分の Google アカウントに限定可能
