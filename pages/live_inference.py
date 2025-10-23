"""
Live Inference 頁面
實時郵件分類功能
"""

import streamlit as st
import time
import random
from typing import Any

def show_page(classifier: Any, data_loader: Any, visualizer: Any, 
              selected_model: str, threshold: float, model_trained: bool):
    """顯示 Live Inference 頁面"""
    
    st.title("🎯 Live Inference")
    st.markdown("輸入郵件內容進行實時垃圾郵件檢測")
    
    if not model_trained:
        st.warning("⚠️ 請先在側邊欄訓練模型才能使用此功能")
        return
    
    # 初始化 session state
    if 'email_text' not in st.session_state:
        st.session_state.email_text = ""
    if 'classification_result' not in st.session_state:
        st.session_state.classification_result = None
    if 'classification_history' not in st.session_state:
        st.session_state.classification_history = []
    if 'force_update' not in st.session_state:
        st.session_state.force_update = False
    
    # 創建兩列布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 郵件輸入")
        
        # 郵件輸入區域
        # 檢查是否需要強制更新
        if 'email_input_key' not in st.session_state:
            st.session_state.email_input_key = 0
            
        # 當需要更新時，改變 key 來強制重新渲染
        if st.session_state.force_update:
            st.session_state.email_input_key += 1
            st.session_state.force_update = False
            
        email_text = st.text_area(
            "輸入郵件內容",
            value=st.session_state.email_text,
            height=200,
            placeholder="請輸入要分類的郵件內容...",
            help="輸入完整的郵件內容，系統將自動進行垃圾郵件檢測",
            key=f"email_input_area_{st.session_state.email_input_key}"
        )
        
        # 同步 session state
        if email_text != st.session_state.email_text:
            st.session_state.email_text = email_text
        
        # 範例郵件按鈕區域
        st.markdown("**快速測試範例:**")
        col_spam, col_ham, col_random = st.columns(3)
        
        # 調試信息（可選）
        if st.checkbox("顯示調試信息", value=False):
            st.write(f"Session State email_text: {len(st.session_state.email_text)} 字符")
            st.write(f"Current email_text: {len(email_text)} 字符")
            st.write(f"Input key: {st.session_state.email_input_key}")
            st.write(f"Force update: {st.session_state.force_update}")
        
        with col_spam:
            if st.button("🚨 垃圾郵件範例", use_container_width=True, key="spam_example_btn"):
                # 使用固定的測試範例確保功能正常
                test_spam = "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
                sample_emails = data_loader.get_sample_emails()
                if sample_emails['spam']:
                    selected_email = random.choice(sample_emails['spam'])
                else:
                    selected_email = test_spam
                    
                st.session_state.email_text = selected_email
                st.session_state.classification_result = None  # 清除之前的結果
                st.session_state.force_update = True
                st.rerun()
        
        with col_ham:
            if st.button("✅ 正常郵件範例", use_container_width=True, key="ham_example_btn"):
                # 使用固定的測試範例確保功能正常
                test_ham = "Hi John, hope you're doing well. Let's meet for coffee tomorrow at 3pm. Looking forward to catching up!"
                sample_emails = data_loader.get_sample_emails()
                if sample_emails['ham']:
                    selected_email = random.choice(sample_emails['ham'])
                else:
                    selected_email = test_ham
                    
                st.session_state.email_text = selected_email
                st.session_state.classification_result = None  # 清除之前的結果
                st.session_state.force_update = True
                st.rerun()
        
        with col_random:
            if st.button("🎲 隨機範例", use_container_width=True, key="random_example_btn"):
                # 從數據集中隨機選擇一封郵件
                data = data_loader.load_raw_dataset()
                if not data.empty:
                    random_email = data.sample(n=1).iloc[0]
                    st.session_state.email_text = random_email['text']
                    st.session_state.classification_result = None  # 清除之前的結果
                    st.session_state.force_update = True
                    st.rerun()
        
        # 顯示當前郵件的基本信息
        if email_text.strip():
            with st.expander("📋 郵件信息", expanded=False):
                word_count = len(email_text.split())
                char_count = len(email_text)
                
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.metric("字數", word_count)
                with info_col2:
                    st.metric("字符數", char_count)
                
                # 顯示郵件預覽
                preview_text = email_text[:200] + "..." if len(email_text) > 200 else email_text
                st.text_area("郵件預覽", value=preview_text, height=80, disabled=True)
        
        # 郵件驗證
        if email_text.strip():
            word_count = len(email_text.split())
            if word_count < 3:
                st.warning("⚠️ 郵件內容太短，可能影響分類準確性")
            elif word_count > 500:
                st.info("ℹ️ 郵件內容較長，處理時間可能稍長")
        
        # 分類按鈕
        st.markdown("---")
        col_classify, col_clear = st.columns([3, 1])
        
        with col_classify:
            classify_button = st.button(
                "🚀 開始分類", 
                use_container_width=True,
                type="primary",
                disabled=not email_text.strip(),
                help="點擊開始分析郵件內容" if email_text.strip() else "請先輸入郵件內容"
            )
        
        with col_clear:
            if st.button("🗑️ 清空", use_container_width=True, help="清空輸入框和結果", key="clear_btn"):
                st.session_state.email_text = ""
                st.session_state.classification_result = None
                st.session_state.force_update = True
                st.rerun()
        
        # 預處理選項
        with st.expander("🔧 預處理選項"):
            show_processed = st.checkbox("顯示預處理後的文本", value=False)
            show_features = st.checkbox("顯示特徵信息", value=False)
    
    with col2:
        st.subheader("📊 分類結果")
        
        # 處理分類按鈕點擊
        if classify_button and email_text.strip():
            # 進行預測
            with st.spinner("正在分析郵件..."):
                result = classifier.predict(email_text, selected_model)
                st.session_state.classification_result = result
                
                # 添加到歷史記錄
                if 'error' not in result:
                    history_item = {
                        'timestamp': time.strftime("%H:%M:%S"),
                        'email_preview': email_text[:50] + "..." if len(email_text) > 50 else email_text,
                        'is_spam': result['spam_probability'] > threshold,
                        'confidence': result['confidence'],
                        'model': selected_model
                    }
                    st.session_state.classification_history.insert(0, history_item)
                    # 保持最近 10 條記錄
                    if len(st.session_state.classification_history) > 10:
                        st.session_state.classification_history = st.session_state.classification_history[:10]
        
        # 顯示分類結果
        if st.session_state.classification_result:
            result = st.session_state.classification_result
            
            if 'error' in result:
                st.error(f"分類時發生錯誤: {result['error']}")
            else:
                # 顯示主要結果
                spam_prob = result['spam_probability']
                is_spam = spam_prob > threshold
                
                # 結果顯示卡片
                if is_spam:
                    st.error("🚨 垃圾郵件")
                else:
                    st.success("✅ 正常郵件")
                
                # 置信度指標
                st.metric(
                    "置信度", 
                    f"{result['confidence']:.1%}",
                    help="模型對此分類結果的信心程度"
                )
                
                # 垃圾郵件機率條形圖
                st.markdown("**垃圾郵件機率**")
                prob_fig = visualizer.create_spam_probability_bar(spam_prob)
                st.plotly_chart(prob_fig, use_container_width=True)
                
                # 詳細信息
                with st.expander("📋 詳細信息", expanded=True):
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.write(f"**模型**: {result['model_used']}")
                        st.write(f"**處理時間**: {result['processing_time']:.2f} ms")
                    
                    with col_info2:
                        st.write(f"**垃圾機率**: {spam_prob:.1%}")
                        st.write(f"**分類閾值**: {threshold:.1%}")
                
                # 預處理文本展示
                if show_processed and 'processed_text' in result:
                    st.markdown("---")
                    st.markdown("**預處理後的文本**")
                    st.text_area(
                        "處理後文本",
                        value=result['processed_text'],
                        height=100,
                        disabled=True,
                        key="processed_text_display"
                    )
                
                # 特徵信息
                if show_features:
                    st.markdown("---")
                    st.markdown("**特徵重要性 (Top 10)**")
                    feature_importance = classifier.get_feature_importance(selected_model, 10)
                    if feature_importance:
                        for i, (feature, importance) in enumerate(feature_importance[:10]):
                            st.write(f"{i+1}. `{feature}`: {importance:.4f}")
                    else:
                        st.info("此模型不支持特徵重要性分析")
        
        elif not email_text.strip():
            st.info("👆 請在左側輸入郵件內容並點擊「開始分類」")
        else:
            st.info("👆 點擊「開始分類」按鈕進行分析")
    
    # 分隔線
    st.markdown("---")
    
    # 範例選擇器
    st.subheader("📚 範例選擇器")
    
    # 獲取更多範例
    data = data_loader.load_raw_dataset()
    
    if not data.empty:
        col_selector1, col_selector2 = st.columns(2)
        
        with col_selector1:
            st.markdown("**垃圾郵件範例**")
            spam_samples = data[data['label'] == 'spam']['text'].head(5).tolist()
            
            for i, sample in enumerate(spam_samples):
                preview = sample[:80] + "..." if len(sample) > 80 else sample
                if st.button(f"🚨 範例 {i+1}: {preview}", key=f"spam_sample_{i}"):
                    st.session_state.email_text = sample
                    st.session_state.classification_result = None
                    st.session_state.force_update = True
                    st.rerun()
        
        with col_selector2:
            st.markdown("**正常郵件範例**")
            ham_samples = data[data['label'] == 'ham']['text'].head(5).tolist()
            
            for i, sample in enumerate(ham_samples):
                preview = sample[:80] + "..." if len(sample) > 80 else sample
                if st.button(f"✅ 範例 {i+1}: {preview}", key=f"ham_sample_{i}"):
                    st.session_state.email_text = sample
                    st.session_state.classification_result = None
                    st.session_state.force_update = True
                    st.rerun()
    
    # 分隔線
    st.markdown("---")
    
    # 批量測試區域
    st.subheader("🔄 批量測試")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("測試 5 個垃圾郵件範例", use_container_width=True):
            sample_emails = data_loader.get_sample_emails()
            if sample_emails['spam']:
                test_batch_emails(classifier, sample_emails['spam'][:5], 
                                selected_model, threshold, visualizer, "垃圾郵件")
    
    with col2:
        if st.button("測試 5 個正常郵件範例", use_container_width=True):
            sample_emails = data_loader.get_sample_emails()
            if sample_emails['ham']:
                test_batch_emails(classifier, sample_emails['ham'][:5], 
                                selected_model, threshold, visualizer, "正常郵件")
    
    with col3:
        if st.button("隨機測試 10 個郵件", use_container_width=True):
            # 從數據集中隨機選擇郵件進行測試
            data = data_loader.load_raw_dataset()
            if not data.empty:
                random_sample = data.sample(n=min(10, len(data)))
                test_results = []
                
                for _, row in random_sample.iterrows():
                    result = classifier.predict(row['text'], selected_model)
                    if 'error' not in result:
                        predicted = "spam" if result['spam_probability'] > threshold else "ham"
                        actual = row['label']
                        correct = predicted == actual
                        
                        test_results.append({
                            'text': row['text'][:50] + "..." if len(row['text']) > 50 else row['text'],
                            'actual': actual,
                            'predicted': predicted,
                            'probability': result['spam_probability'],
                            'correct': correct
                        })
                
                if test_results:
                    st.subheader("隨機測試結果")
                    results_df = st.dataframe(test_results, use_container_width=True)
                    
                    accuracy = sum(1 for r in test_results if r['correct']) / len(test_results)
                    st.metric("準確率", f"{accuracy:.1%}")
    
    # 分類歷史記錄
    if st.session_state.classification_history:
        st.markdown("---")
        st.subheader("📋 分類歷史")
        
        # 清空歷史按鈕
        col_history_title, col_clear_history = st.columns([4, 1])
        with col_clear_history:
            if st.button("🗑️ 清空歷史", key="clear_history"):
                st.session_state.classification_history = []
                st.rerun()
        
        # 顯示歷史記錄
        for i, item in enumerate(st.session_state.classification_history):
            with st.expander(f"{item['timestamp']} - {'🚨 垃圾郵件' if item['is_spam'] else '✅ 正常郵件'} ({item['confidence']:.1%})"):
                st.write(f"**郵件內容**: {item['email_preview']}")
                st.write(f"**使用模型**: {item['model']}")
                st.write(f"**分類時間**: {item['timestamp']}")

def test_batch_emails(classifier, emails, model_name, threshold, visualizer, email_type):
    """批量測試郵件"""
    st.subheader(f"{email_type}測試結果")
    
    results = []
    progress_bar = st.progress(0)
    
    for i, email in enumerate(emails):
        result = classifier.predict(email, model_name)
        if 'error' not in result:
            is_spam = result['spam_probability'] > threshold
            results.append({
                '郵件內容': email[:50] + "..." if len(email) > 50 else email,
                '預測結果': "垃圾郵件" if is_spam else "正常郵件",
                '置信度': f"{result['confidence']:.1%}",
                '垃圾郵件機率': f"{result['spam_probability']:.1%}"
            })
        
        progress_bar.progress((i + 1) / len(emails))
    
    if results:
        st.dataframe(results, use_container_width=True)
        
        # 統計信息
        spam_predictions = sum(1 for r in results if r['預測結果'] == "垃圾郵件")
        st.write(f"**預測為垃圾郵件**: {spam_predictions}/{len(results)}")
        st.write(f"**預測為正常郵件**: {len(results) - spam_predictions}/{len(results)}")
    
    progress_bar.empty()