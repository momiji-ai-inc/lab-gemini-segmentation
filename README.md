## 事前準備

### 仮装環境
```sh
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### その他準備
- `/images`に解析対象の画像を配置
- `.env`を作成して、API KEYをセット

## 実行方法
- image: 画像パス str
- query: セグメント対象物 str
- alpha: マスクの透明度(任意) float
```bash
python src/main.py --image ./images/cat.jpg --query cat
```

## Gemini API レスポンス仕様

レスポンス形式は、画像内の物体ごとに以下の情報を持つJSONリスト

- `box_2d`: [y0, x0, y1, x1]（画像内の矩形領域、0-1000で正規化された座標）
- `mask`: base64エンコードPNG（box領域の確率マップ、0-255）
- `label`: 物体の説明ラベル（日本語含む）

例:
```json
[
  {
    "box_2d": [100, 200, 300, 400],
    "mask": "iVBORw0KGgoAAAANSUhEUgAA...",
    "label": "猫"
  },
  ...
]
```

## 注意点
- セグメンテーション結果の画像にラベル表記をする際、日本語だと▫️表示になる場合があります。デフォルトはmacOS標準の日本語フォント`/System/Library/Fonts/Hiragino Sans GB.ttc`をロードする設計ですので、文字化けが生じた場合は必要に応じて変更してください。
- `gemini-2.5-pro`を利用する場合は、`thinking_budget`を0より大きく設定(128~32768)する必要があります。(試した感じ、2.5-flash-liteが良さそう)