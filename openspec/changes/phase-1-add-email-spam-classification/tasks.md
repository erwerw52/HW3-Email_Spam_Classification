# Phase 1 實施任務清單

## 1. Phase 1-A: 基礎架構設置
- [ ] 1.1 建立項目結構和依賴管理
  - 創建 Streamlit 應用項目結構
  - 配置 requirements.txt (包含 streamlit, scikit-learn, pandas 等)
  - 設置開發環境配置
- [ ] 1.2 設計數據模型
  - 定義郵件實體模型
  - 設計分類結果模型
  - 創建 SQLite 數據庫結構
- [ ] 1.3 建立 Streamlit 應用基礎架構
  - 創建主應用文件 (app.py)
  - 設置頁面配置和導航
  - 實現基礎 UI 組件

## 2. Phase 1-B: 數據處理和可視化模塊
- [ ] 2.1 數據載入和預處理
  - 垃圾郵件數據集載入 (CSV 格式)
  - 文本清理和標準化
  - HTML 標籤移除和特殊字符處理
  - 數據集分割 (訓練/測試)
- [ ] 2.2 特徵提取和分析
  - TF-IDF 向量化實現
  - N-gram 特徵提取 (1-gram, 2-gram)
  - Top Tokens 統計分析
  - 詞頻分布計算
- [ ] 2.3 數據可視化功能
  - 類別分布圖表 (Plotly Bar Chart)
  - Top Tokens 橫向條形圖
  - 詞雲圖生成 (WordCloud)
  - 特徵重要性可視化

## 3. Phase 1-C: 多模型機器學習分類器
- [ ] 3.1 實現多種分類模型
  - Naive Bayes 分類器
  - SVM 分類器 (線性和 RBF 核)
  - Random Forest 分類器
  - Logistic Regression 分類器
  - 統一的模型接口設計
- [ ] 3.2 模型評估和性能分析
  - 交叉驗證實現
  - ROC 曲線和 AUC 計算
  - Precision-Recall 曲線
  - 混淆矩陣生成
  - 性能指標比較表格
- [ ] 3.3 模型持久化和管理
  - 多模型序列化和載入
  - 模型版本管理
  - Streamlit 緩存優化

## 4. Phase 1-D: Streamlit 多頁面界面實現
- [ ] 4.1 Live Inference 頁面
  - 實現郵件輸入文本區域
  - 範例郵件按鈕 (垃圾郵件/正常郵件)
  - 實時分類結果顯示
  - 置信度機率條形圖
  - 預處理文本展示功能
- [ ] 4.2 Model Performance 頁面
  - 多模型性能比較表格
  - ROC 曲線圖表 (使用 Plotly)
  - Precision-Recall 曲線圖表
  - 互動式混淆矩陣
  - 模型選擇下拉選單
- [ ] 4.3 Data Visualization 頁面
  - 數據概覽指標卡片
  - 類別分布圖表
  - Top Tokens 分析 (垃圾郵件 vs 正常郵件)
  - 詞雲圖生成
  - 特徵重要性可視化

## 5. Phase 1-E: 互動功能和側邊欄
- [ ] 5.1 側邊欄配置界面
  - 模型選擇下拉選單
  - 模型參數調整滑桿 (SVM C 參數、核函數等)
  - 閾值調整功能
  - 數據集選擇選項
- [ ] 5.2 互動式圖表功能
  - Plotly 互動圖表實現
  - 圖表縮放和懸停效果
  - 動態數據更新
  - 圖表匯出功能
- [ ] 5.3 用戶體驗優化
  - 載入狀態指示器
  - 錯誤處理和友好提示
  - 響應式布局設計
  - 快捷鍵支持

## 6. Phase 1-F: Streamlit Cloud 部署
- [ ] 6.1 部署準備
  - 完整的 requirements.txt 文件 (包含所有可視化庫)
  - .streamlit/config.toml 配置優化
  - 數據文件和模型文件準備
  - 環境變量和秘密配置
- [ ] 6.2 Streamlit Cloud 部署
  - GitHub 倉庫連接和配置
  - 部署設置和資源配置
  - 線上功能完整性測試
  - 性能監控設置
- [ ] 6.3 部署優化和維護
  - 緩存策略優化 (@st.cache_data, @st.cache_resource)
  - 記憶體使用優化
  - 用戶使用指南和 README
  - 持續集成和自動部署