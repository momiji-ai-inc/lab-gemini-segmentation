## 事前準備

### 仮装環境
```sh
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 画像の保存
`/images`に解析対象の画像を配置

## 実行方法
- image: 画像パス str
- query: セグメント対象物 str
- alpha: マスクの透明度(任意) float
```bash
python src/main.py --image ./images/cat.jpg --query cat
```