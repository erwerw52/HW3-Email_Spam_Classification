"""
垃圾郵件分類系統 - Phase 1
主應用文件
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# 導入自定義模塊
from src.config import APP_CONFIG
from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor
from src.classifier import MultiModelClassifier
from src.visualizations import Visualizer

# 導入頁面模塊
from pages import live_inference, model_performance, data_visualization

def main():
    """主應用函數"""
    
    # 頁面配置
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon=APP_CONFIG['icon'],
        layout=APP_CONFIG['layout'],
        initial_sidebar_state="expanded"
    )
    
    # 初始化 session state（必須在最開始）
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'fast_mode' not in st.session_state:
        st.session_state.fast_mode = True
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    
    # 自定義 CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    
    /* 注意：不要隱藏所有 selectbox，只隱藏特定的導航相關元素 */
    
    /* 隱藏頁面頂部的導航欄 */
    .css-1d391kg, .css-1rs6os, .css-17ziqus {
        display: none !important;
    }
    
    /* 隱藏多頁面應用的頁面選擇器 */
    .css-1kyxreq {
        display: none !important;
    }
    
    /* 隱藏 Streamlit 的頁面導航 */
    section[data-testid="stSidebar"] > div:first-child > div:first-child {
        display: none !important;
    }
    
    /* 隱藏側邊欄頂部的應用信息 */
    .css-1lcbmhc, .css-1outpf7 {
        display: none !important;
    }
    
    /* 更精確地隱藏多頁面導航區域 */
    .stApp > header {
        display: none !important;
    }
    
    /* 隱藏側邊欄中的頁面導航按鈕 */
    .css-1544g2n, .css-1v0mbdj {
        display: none !important;
    }
    
    /* 隱藏 Streamlit 默認的頁面標題區域 */
    .css-18e3th9, .css-1d391kg {
        display: none !important;
    }
    
    /* 隱藏多頁面應用的導航欄 */
    div[data-testid="stSidebarNav"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 主標題
    st.markdown('<h1 class="main-header">📧 垃圾郵件分類系統</h1>', unsafe_allow_html=True)
    
    # 初始化基礎組件
    data_loader = DataLoader()
    preprocessor = TextPreprocessor()
    visualizer = Visualizer()
    
    # 側邊欄配置
    st.sidebar.markdown('<h2 class="sidebar-header">🔧 系統配置</h2>', unsafe_allow_html=True)
    
    # 頁面選擇
    page = st.sidebar.radio(
        "選擇功能頁面",
        ["🎯 Live Inference", "📊 Model Performance", "📈 Data Visualization"],
        help="選擇要使用的功能頁面"
    )
    
    # 模型配置區域
    st.sidebar.markdown("### 模型設置")
    
    # 快速訓練模式（移到外面）
    fast_mode = st.sidebar.checkbox(
        "快速訓練模式",
        value=True,
        help="啟用快速模式可大幅減少訓練時間，但可能略微降低準確率"
    )
    
    # 數據集大小選項（移到外面）
    dataset_options = ["完整數據集", "50% 數據集", "25% 數據集", "10% 數據集"]
    dataset_size = st.sidebar.selectbox(
        "數據集大小",
        options=dataset_options,
        index=1 if fast_mode else 0,
        help="選擇用於訓練的數據集大小，較小的數據集訓練更快",
        key="dataset_size_selector"
    )
    
    # 高級設置
    with st.sidebar.expander("高級設置"):
        # 分類閾值
        threshold = st.slider(
            "分類閾值",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="調整垃圾郵件分類的閾值"
        )
        
        # 特徵數量
        max_features = st.slider(
            "最大特徵數",
            min_value=1000,
            max_value=10000,
            value=3000 if fast_mode else 5000,
            step=500,
            help="TF-IDF 向量化的最大特徵數量"
        )
        
        # 是否移除停用詞
        remove_stopwords = st.checkbox(
            "移除停用詞",
            value=True,
            help="在預處理時是否移除英文停用詞"
        )
        
        # 調試信息（可選）
        if st.checkbox("顯示調試信息", value=False, key="debug_info"):
            st.write(f"選擇的數據集大小: {dataset_size}")
            st.write(f"快速模式: {fast_mode}")
            st.write(f"可用選項: {dataset_options}")
    
    # 根據快速模式設置初始化分類器
    if st.session_state.classifier is None or st.session_state.fast_mode != fast_mode:
        st.session_state.classifier = MultiModelClassifier(fast_mode=fast_mode)
        st.session_state.fast_mode = fast_mode
        st.session_state.model_trained = False  # 重置訓練狀態
    
    classifier = st.session_state.classifier
    
    # 模型選擇
    if st.session_state.model_trained and hasattr(classifier, 'get_trained_models'):
        available_models = classifier.get_trained_models()
        if available_models:
            selected_model = st.sidebar.selectbox(
                "選擇分類模型",
                available_models,
                help="選擇用於分類的機器學習模型"
            )
        else:
            st.sidebar.error("沒有可用的訓練模型")
            selected_model = APP_CONFIG['model_names'][0]
    else:
        selected_model = st.sidebar.selectbox(
            "選擇分類模型",
            APP_CONFIG['model_names'],
            help="選擇用於分類的機器學習模型（需要先訓練模型）",
            disabled=True
        )
    
    # 數據載入狀態
    st.sidebar.markdown("### 系統狀態")
    
    # 載入數據
    with st.spinner("載入數據中..."):
        data = data_loader.load_raw_dataset()
        dataset_info = data_loader.get_dataset_info()
    
    if not data.empty:
        st.sidebar.success("✅ 數據載入成功")
        st.sidebar.info(f"總郵件數: {dataset_info.get('total_messages', 0)}")
        st.sidebar.info(f"垃圾郵件: {dataset_info.get('spam_count', 0)}")
        st.sidebar.info(f"正常郵件: {dataset_info.get('ham_count', 0)}")
    else:
        st.sidebar.error("❌ 數據載入失敗")
        st.error("無法載入數據集，請檢查數據文件是否存在。")
        return
    
    # 訓練模型按鈕
    train_button_col1, train_button_col2 = st.sidebar.columns([3, 1])
    
    with train_button_col1:
        train_clicked = st.button("🚀 訓練模型", help="訓練所有機器學習模型")
    
    with train_button_col2:
        if fast_mode:
            st.caption("⚡ 快速模式")
        else:
            st.caption("🎯 標準模式")
    
    # 顯示預估訓練時間
    time_estimates = {
        ("完整數據集", True): "1-2 分鐘",
        ("完整數據集", False): "3-8 分鐘", 
        ("50% 數據集", True): "30-60 秒",
        ("50% 數據集", False): "1-3 分鐘",
        ("25% 數據集", True): "15-30 秒",
        ("25% 數據集", False): "30-90 秒",
        ("10% 數據集", True): "5-15 秒",
        ("10% 數據集", False): "15-30 秒"
    }
    
    estimated_time = time_estimates.get((dataset_size, fast_mode), "未知")
    st.sidebar.info(f"⏱️ 預估訓練時間: {estimated_time}")
    
    if train_clicked:
        try:
            # 準備數據
            with st.spinner("準備數據中..."):
                train_df, test_df = data_loader.split_data()
                
                # 根據選擇的數據集大小進行採樣
                if dataset_size != "完整數據集":
                    size_map = {"50% 數據集": 0.5, "25% 數據集": 0.25, "10% 數據集": 0.1}
                    sample_ratio = size_map[dataset_size]
                    
                    train_df = train_df.sample(frac=sample_ratio, random_state=42)
                    test_df = test_df.sample(frac=sample_ratio, random_state=42)
                    
                    st.sidebar.info(f"使用 {len(train_df)} 個訓練樣本，{len(test_df)} 個測試樣本")
                
                # 預處理文本
                train_texts = [preprocessor.preprocess_text(text, remove_stopwords) 
                              for text in train_df['text']]
                test_texts = [preprocessor.preprocess_text(text, remove_stopwords) 
                             for text in test_df['text']]
                
                # 訓練向量化器
                preprocessor.fit_vectorizers(train_texts, max_features)
                
                # 提取特徵
                X_train = preprocessor.extract_tfidf_features(train_texts)
                X_test = preprocessor.extract_tfidf_features(test_texts)
                
                y_train = train_df['label'].values
                y_test = test_df['label'].values
            
            st.sidebar.success("✅ 數據準備完成")
            
            # 訓練模型
            classifier.preprocessor = preprocessor
            classifier.train_models(X_train, y_train, X_test, y_test)
            
            # 保存模型
            classifier.save_models()
            
            st.session_state.model_trained = True
            st.sidebar.success("✅ 模型訓練完成")
            
        except Exception as e:
            st.sidebar.error(f"❌ 模型訓練失敗: {str(e)}")
            st.error(f"詳細錯誤信息: {str(e)}")
    
    # 嘗試載入已訓練的模型
    if not st.session_state.model_trained:
        if classifier.load_models():
            st.session_state.model_trained = True
            st.sidebar.success("✅ 已載入預訓練模型")
    
    # 顯示模型狀態
    if st.session_state.model_trained:
        st.sidebar.success("🤖 模型已就緒")
        
        # 顯示已訓練的模型
        if hasattr(classifier, 'get_trained_models'):
            trained_models = classifier.get_trained_models()
            model_status = classifier.get_model_status()
            
            with st.sidebar.expander("模型狀態詳情"):
                for model_name, is_trained in model_status.items():
                    if is_trained:
                        st.success(f"✅ {model_name}")
                    else:
                        st.error(f"❌ {model_name}")
                
                st.write(f"**可用模型**: {len(trained_models)}/{len(APP_CONFIG['model_names'])}")
    else:
        st.sidebar.warning("⚠️ 請先訓練模型")
    
    # 頁面路由
    if page == "🎯 Live Inference":
        live_inference.show_page(
            classifier=classifier,
            data_loader=data_loader,
            visualizer=visualizer,
            selected_model=selected_model,
            threshold=threshold,
            model_trained=st.session_state.model_trained
        )
    elif page == "📊 Model Performance":
        model_performance.show_page(
            classifier=classifier,
            visualizer=visualizer,
            model_trained=st.session_state.model_trained
        )
    elif page == "📈 Data Visualization":
        data_visualization.show_page(
            data=data,
            data_loader=data_loader,
            preprocessor=preprocessor,
            visualizer=visualizer
        )
    
    # 頁腳
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>垃圾郵件分類系統 - Phase 1 | 
            基於 Streamlit 和 scikit-learn 構建 | 
            © 2025</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()