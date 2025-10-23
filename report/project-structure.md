# Phase 1 項目結構規劃

## 📁 推薦的項目目錄結構

```
spam-email-classifier/
├── 📄 app.py                          # 主應用入口文件
├── 📁 pages/                          # Streamlit 頁面模塊
│   ├── 📄 __init__.py
│   ├── 📄 live_inference.py           # 🎯 實時推理頁面
│   ├── 📄 model_performance.py        # 📊 模型性能頁面
│   └── 📄 data_visualization.py       # 📈 數據可視化頁面
├── 📁 src/                           # 核心業務邏輯
│   ├── 📄 __init__.py
│   ├── 📄 classifier.py              # 多模型分類器
│   ├── 📄 data_loader.py             # 數據載入器
│   ├── 📄 preprocessing.py           # 文本預處理
│   ├── 📄 visualizations.py          # 可視化函數
│   └── 📄 utils.py                   # 工具函數
├── 📁 data/                          # 數據文件
│   ├── 📄 spam_dataset.csv           # 垃圾郵件數據集
│   └── 📄 sample_emails.json         # 範例郵件
├── 📁 models/                        # 模型文件
│   ├── 📄 trained_models.pkl         # 預訓練模型
│   ├── 📄 vectorizer.pkl             # TF-IDF 向量化器
│   └── 📄 model_config.json          # 模型配置
├── 📁 .streamlit/                    # Streamlit 配置
│   └── 📄 config.toml                # 應用配置
├── 📁 assets/                        # 靜態資源
│   ├── 📄 logo.png                   # 應用圖標
│   └── 📄 style.css                  # 自定義樣式
├── 📁 tests/                         # 測試文件
│   ├── 📄 __init__.py
│   ├── 📄 test_classifier.py         # 分類器測試
│   └── 📄 test_preprocessing.py      # 預處理測試
├── 📄 requirements.txt               # Python 依賴
├── 📄 README.md                      # 項目說明
├── 📄 .gitignore                     # Git 忽略文件
└── 📄 setup.py                       # 包安裝配置
```

## 📋 核心文件說明

### 🎯 主應用文件 (`app.py`)
- Streamlit 應用的入口點
- 頁面路由和導航邏輯
- 側邊欄配置和全局設置
- 數據和模型的緩存載入

### 📄 頁面模塊 (`pages/`)

#### `live_inference.py`
- 郵件輸入界面
- 實時分類功能
- 結果展示和可視化
- 範例郵件按鈕

#### `model_performance.py`
- ROC 曲線圖表
- Precision-Recall 曲線
- 混淆矩陣展示
- 模型比較表格

#### `data_visualization.py`
- 數據概覽統計
- 類別分布圖表
- Top Tokens 分析
- 詞雲圖生成

### 🔧 核心業務邏輯 (`src/`)

#### `classifier.py`
```python
class MultiModelClassifier:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        
    def train_models(self, X, y):
        # 訓練多種模型
        
    def predict(self, text, model_name):
        # 預測分類結果
        
    def get_model_comparison(self):
        # 返回模型比較數據
```

#### `data_loader.py`
```python
class DataLoader:
    def load_spam_dataset(self):
        # 載入垃圾郵件數據集
        
    def get_sample_emails(self):
        # 獲取範例郵件
```

#### `preprocessing.py`
```python
def clean_text(text):
    # 文本清理和標準化
    
def extract_features(text):
    # 特徵提取
```

#### `visualizations.py`
```python
def create_roc_curve(y_true, y_scores):
    # 創建 ROC 曲線
    
def create_confusion_matrix(y_true, y_pred):
    # 創建混淆矩陣
    
def create_wordcloud(text_data):
    # 創建詞雲圖
```

## 📦 依賴管理 (`requirements.txt`)

```txt
# 核心框架
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0

# 機器學習
scikit-learn>=1.3.0
joblib>=1.3.0

# 數據可視化
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
wordcloud>=1.9.0

# 文本處理
nltk>=3.8.0
beautifulsoup4>=4.12.0

# 工具庫
python-dotenv>=1.0.0
```

## ⚙️ Streamlit 配置 (`.streamlit/config.toml`)

```toml
[global]
developmentMode = false

[server]
runOnSave = true
port = 8501

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## 🚀 快速開始指南

### 1. 環境設置
```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

### 2. 數據準備
```bash
# 下載垃圾郵件數據集 (例如 SMS Spam Collection)
# 放置到 data/spam_dataset.csv
```

### 3. 運行應用
```bash
streamlit run app.py
```

### 4. 部署到 Streamlit Cloud
1. 推送代碼到 GitHub
2. 連接 Streamlit Cloud
3. 配置部署設置
4. 啟動應用

## 📝 開發順序建議

### Phase 1-A: 基礎設置
1. 創建項目結構
2. 設置 `requirements.txt`
3. 創建基本的 `app.py`

### Phase 1-B: 數據處理
1. 實現 `data_loader.py`
2. 實現 `preprocessing.py`
3. 準備數據集

### Phase 1-C: 模型開發
1. 實現 `classifier.py`
2. 訓練和保存模型
3. 測試分類功能

### Phase 1-D: 界面開發
1. 實現 `live_inference.py`
2. 實現 `model_performance.py`
3. 實現 `data_visualization.py`

### Phase 1-E: 優化和測試
1. 添加互動功能
2. 性能優化
3. 用戶體驗改進

### Phase 1-F: 部署
1. 準備部署配置
2. 部署到 Streamlit Cloud
3. 測試和維護

這個結構設計確保了代碼的模塊化、可維護性和可擴展性，同時符合 Streamlit 應用的最佳實踐。