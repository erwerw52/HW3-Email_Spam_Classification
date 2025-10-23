# Phase 1: 電子郵件垃圾分類系統設計

## Context
Phase 1 專注於建立電子郵件垃圾分類系統的核心基礎設施。這個階段的目標是創建一個可工作的最小可行產品 (MVP)，提供基本的垃圾郵件檢測功能，為後續階段奠定堅實的基礎。

## Goals / Non-Goals

### Goals
- 建立專業級的 Streamlit 應用架構
- 實現高準確率的垃圾郵件分類功能 (準確率 > 95%)
- 提供多頁面互動式 Web 界面
- 支持實時推理和模型性能分析
- 提供豐富的數據可視化功能
- 支持多種模型比較和參數調整
- 支持 Streamlit Cloud 部署

### Non-Goals (留待後續 Phase)
- 深度學習模型 (Phase 2)
- 批量處理功能 (Phase 2)
- 多語言支持 (Phase 2)
- 複雜的用戶管理系統 (Phase 3)
- 企業級部署架構 (Phase 4)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Multi-Page App                 │
├─────────────────┬─────────────────┬─────────────────────────┤
│ Live Inference  │ Model Performance│   Data Visualization    │
│                 │                 │                         │
│ - 郵件輸入界面   │ - ROC 曲線      │ - 類別分布圖            │
│ - 實時分類結果   │ - PR 曲線       │ - Top Tokens 分析       │
│ - 置信度顯示     │ - 混淆矩陣      │ - 特徵重要性圖表        │
│ - 預處理文本展示 │ - 性能指標      │ - 詞雲圖               │
└─────────────────┴─────────────────┴─────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Classification Service                    │
├─────────────────┬─────────────────┬─────────────────────────┤
│ Feature Extract │   ML Models     │    Visualization        │
│                 │                 │                         │
│ - TF-IDF        │ - Naive Bayes   │ - Plotly Charts        │
│ - N-grams       │ - SVM           │ - Matplotlib Plots     │
│ - 詞頻統計       │ - Random Forest │ - Seaborn Heatmaps     │
│ - 文本清理       │ - Logistic Reg  │ - WordCloud            │
└─────────────────┴─────────────────┴─────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                            │
│  - 訓練數據集 (CSV)  - 預訓練模型 (PKL)  - 配置文件 (JSON)   │
└─────────────────────────────────────────────────────────────┘
```

## Decisions

### Decision 1: 使用 Streamlit 作為 Web 框架
**理由**: 
- 快速原型開發和部署
- 內建的 UI 組件和互動功能
- 簡單的狀態管理
- 優秀的數據科學生態系統集成
- 免費的 Streamlit Cloud 部署

**替代方案**: Flask + HTML/CSS, FastAPI + React
**選擇原因**: Streamlit 適合快速構建 ML 應用原型

### Decision 2: Phase 1 使用傳統 ML 算法
**理由**:
- 快速實現和部署
- 較低的計算資源需求
- 易於理解和調試
- 為後續深度學習模型建立基準

**替代方案**: 直接使用深度學習模型
**選擇原因**: 分階段實施，先建立基礎設施

### Decision 3: 分層架構設計
```
UI Layer (Streamlit)
    ↓
Service Layer (Business Logic)
    ↓
Data Layer (SQLite + 模型文件)
```

### Decision 4: 特徵提取策略
- **文本特徵**: TF-IDF, N-grams
- **郵件頭特徵**: 發送者信息, 主題行模式
- **結構特徵**: HTML/文本比例, 鏈接數量

### Decision 5: 數據存儲策略
**理由**: 
- SQLite 用於本地開發和小規模部署
- 分類歷史記錄存儲在本地數據庫
- 模型文件使用 Pickle/Joblib 序列化

**替代方案**: PostgreSQL, 雲數據庫
**選擇原因**: 簡化部署，適合 Streamlit Cloud

### Decision 6: Streamlit Cloud 部署
**理由**:
- 免費託管服務
- 與 GitHub 集成
- 自動部署和更新
- 適合原型和演示

## Component Details

### 1. Streamlit 多頁面應用結構
```python
# app.py - 主應用文件
import streamlit as st
from pages import live_inference, model_performance, data_visualization
from src.classifier import MultiModelClassifier
from src.data_loader import DataLoader

def main():
    st.set_page_config(
        page_title="垃圾郵件分類系統 - Phase 4",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 側邊欄配置
    st.sidebar.title("🔧 模型配置")
    
    # 模型選擇
    model_type = st.sidebar.selectbox(
        "選擇模型",
        ["Naive Bayes", "SVM", "Random Forest", "Logistic Regression"]
    )
    
    # 參數調整
    if model_type == "SVM":
        c_param = st.sidebar.slider("C 參數", 0.1, 10.0, 1.0)
        kernel = st.sidebar.selectbox("核函數", ["linear", "rbf", "poly"])
    
    # 頁面導航
    page = st.sidebar.radio(
        "選擇頁面",
        ["🎯 Live Inference", "📊 Model Performance", "📈 Data Visualization"]
    )
    
    # 載入數據和模型
    @st.cache_data
    def load_data():
        return DataLoader().load_spam_dataset()
    
    @st.cache_resource
    def load_models():
        return MultiModelClassifier()
    
    data = load_data()
    classifier = load_models()
    
    # 頁面路由
    if page == "🎯 Live Inference":
        live_inference.show_page(classifier, model_type)
    elif page == "📊 Model Performance":
        model_performance.show_page(classifier, data)
    elif page == "📈 Data Visualization":
        data_visualization.show_page(data)
```

### 2. Email Classifier Service
```python
class EmailClassifier:
    def __init__(self, model_path: str):
        self.feature_extractor = FeatureExtractor()
        self.model = self.load_model(model_path)
    
    def classify(self, email: EmailData) -> ClassificationResult:
        features = self.feature_extractor.extract(email)
        prediction = self.model.predict(features)
        return ClassificationResult(
            is_spam=prediction.is_spam,
            confidence=prediction.confidence,
            features_used=features.keys()
        )
```

### 2. Feature Extraction Pipeline
```python
class FeatureExtractor:
    def extract(self, email: EmailData) -> Dict[str, float]:
        text_features = self.extract_text_features(email.content)
        header_features = self.extract_header_features(email.headers)
        structural_features = self.extract_structural_features(email)
        
        return {**text_features, **header_features, **structural_features}
```

### 3. Live Inference 頁面設計
```python
# pages/live_inference.py
def show_page(classifier, model_type):
    st.title("🎯 Live Inference")
    st.markdown("輸入郵件內容進行實時垃圾郵件檢測")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 郵件輸入區域
        email_text = st.text_area(
            "輸入郵件內容",
            height=200,
            placeholder="請輸入要分類的郵件內容..."
        )
        
        # 使用範例按鈕
        if st.button("使用垃圾郵件範例"):
            email_text = "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005..."
        
        if st.button("使用正常郵件範例"):
            email_text = "Hi John, hope you're doing well. Let's meet for coffee tomorrow..."
    
    with col2:
        if email_text:
            # 預處理展示
            with st.expander("📝 預處理文本"):
                processed_text = classifier.preprocess_text(email_text)
                st.text(processed_text)
            
            # 分類結果
            result = classifier.predict(email_text, model_type)
            
            # 結果顯示
            if result.is_spam:
                st.error("🚨 垃圾郵件")
                st.metric("置信度", f"{result.confidence:.1%}")
            else:
                st.success("✅ 正常郵件")
                st.metric("置信度", f"{result.confidence:.1%}")
            
            # 機率條形圖
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=[result.spam_probability],
                y=['Spam Probability'],
                orientation='h',
                marker_color='red' if result.is_spam else 'green'
            ))
            fig.update_layout(
                xaxis=dict(range=[0, 1]),
                height=100,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
```

### 4. Model Performance 頁面設計
```python
# pages/model_performance.py
def show_page(classifier, data):
    st.title("📊 Model Performance")
    
    # 模型比較表格
    st.subheader("模型比較")
    performance_df = classifier.get_model_comparison()
    st.dataframe(performance_df, use_container_width=True)
    
    # ROC 和 PR 曲線
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC 曲線")
        roc_fig = classifier.plot_roc_curves()
        st.plotly_chart(roc_fig, use_container_width=True)
    
    with col2:
        st.subheader("Precision-Recall 曲線")
        pr_fig = classifier.plot_precision_recall_curves()
        st.plotly_chart(pr_fig, use_container_width=True)
    
    # 混淆矩陣
    st.subheader("混淆矩陣")
    selected_model = st.selectbox("選擇模型", classifier.model_names)
    cm_fig = classifier.plot_confusion_matrix(selected_model)
    st.plotly_chart(cm_fig, use_container_width=True)
```

### 5. Data Visualization 頁面設計
```python
# pages/data_visualization.py
def show_page(data):
    st.title("📈 Data Visualization")
    
    # 數據概覽
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("總郵件數", len(data))
    with col2:
        spam_count = len(data[data['label'] == 'spam'])
        st.metric("垃圾郵件", spam_count)
    with col3:
        ham_count = len(data[data['label'] == 'ham'])
        st.metric("正常郵件", ham_count)
    
    # 類別分布圖
    st.subheader("類別分布")
    class_dist_fig = create_class_distribution_chart(data)
    st.plotly_chart(class_dist_fig, use_container_width=True)
    
    # Top Tokens 分析
    st.subheader("Top Tokens by Class")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**垃圾郵件 Top Tokens**")
        spam_tokens_fig = create_top_tokens_chart(data, 'spam')
        st.plotly_chart(spam_tokens_fig, use_container_width=True)
    
    with col2:
        st.write("**正常郵件 Top Tokens**")
        ham_tokens_fig = create_top_tokens_chart(data, 'ham')
        st.plotly_chart(ham_tokens_fig, use_container_width=True)
    
    # 詞雲圖
    st.subheader("詞雲圖")
    wordcloud_type = st.radio("選擇類型", ["垃圾郵件", "正常郵件"])
    wordcloud_fig = create_wordcloud(data, wordcloud_type)
    st.pyplot(wordcloud_fig)
```

## Data Models

### Email Data Model
```python
@dataclass
class EmailData:
    subject: str
    content: str
    sender: str
    headers: Dict[str, str]
    timestamp: datetime
```

### Classification Result Model
```python
@dataclass
class ClassificationResult:
    is_spam: bool
    confidence: float
    features_used: List[str]
    processing_time: float
    model_version: str
```

## Risks / Trade-offs

### Risk 1: 模型準確率不足
**緩解措施**: 
- 使用多個模型集成
- 建立完善的評估指標
- 準備回退到規則引擎

### Risk 2: 性能瓶頸
**緩解措施**:
- 使用 Streamlit 緩存機制 (@st.cache_data)
- 優化模型加載和預測
- 監控應用響應時間

### Risk 3: 特徵工程複雜度
**緩解措施**:
- 從簡單特徵開始
- 建立特徵重要性分析
- 模塊化特徵提取器

## Migration Plan

### Phase 1 部署步驟:
1. 設置開發環境
2. 實施核心功能
3. 本地 Streamlit 測試
4. 準備 requirements.txt 和配置文件
5. 部署到 Streamlit Cloud
6. 生產環境測試和優化

### 回退計劃:
- 保持簡單的規則引擎作為備用
- 版本化部署支持快速回退
- 監控告警確保問題早期發現

## Open Questions

1. **訓練數據來源**: 如何獲取高質量的標註數據？
2. **模型更新頻率**: 多久重新訓練一次模型？
3. **多語言處理**: Phase 1 是否需要基礎的多語言支持？
4. **性能目標**: 具體的響應時間和吞吐量要求？

## Phase 2 準備

Phase 1 完成後，為 Phase 2 準備的基礎設施:
- 模型版本管理系統
- 特徵存儲和管理
- A/B 測試框架
- 更詳細的監控指標