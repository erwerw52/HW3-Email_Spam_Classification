"""
Data Visualization 頁面
數據可視化和探索性數據分析
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any

def show_page(data: pd.DataFrame, data_loader: Any, preprocessor: Any, visualizer: Any):
    """顯示 Data Visualization 頁面"""
    
    st.title("📈 Data Visualization")
    st.markdown("探索和可視化垃圾郵件數據集的特徵和分布")
    
    if data.empty:
        st.error("無法載入數據集")
        return
    
    # 數據概覽
    st.subheader("📊 數據集概覽")
    
    dataset_info = data_loader.get_dataset_info()
    
    # 創建指標卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "總郵件數", 
            f"{dataset_info.get('total_messages', 0):,}",
            help="數據集中的總郵件數量"
        )
    
    with col2:
        st.metric(
            "垃圾郵件", 
            f"{dataset_info.get('spam_count', 0):,}",
            delta=f"{dataset_info.get('spam_percentage', 0):.1f}%",
            help="垃圾郵件數量和比例"
        )
    
    with col3:
        st.metric(
            "正常郵件", 
            f"{dataset_info.get('ham_count', 0):,}",
            delta=f"{dataset_info.get('ham_percentage', 0):.1f}%",
            help="正常郵件數量和比例"
        )
    
    with col4:
        st.metric(
            "平均長度", 
            f"{dataset_info.get('avg_message_length', 0):.0f}",
            help="郵件平均字符長度"
        )
    
    # 類別分布圖
    st.subheader("📊 類別分布")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        class_dist_fig = visualizer.create_class_distribution_chart(data)
        st.plotly_chart(class_dist_fig, use_container_width=True)
    
    with col2:
        # 餅圖
        import plotly.express as px
        class_counts = data['label'].value_counts()
        pie_fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title="類別比例分布",
            color_discrete_map={'ham': '#4ECDC4', 'spam': '#FF6B6B'}
        )
        st.plotly_chart(pie_fig, use_container_width=True)
    
    # 文本長度分析
    st.subheader("📏 文本長度分析")
    
    text_length_fig = visualizer.create_text_length_distribution(data)
    st.plotly_chart(text_length_fig, use_container_width=True)
    
    # 文本長度統計
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**垃圾郵件文本統計**")
        spam_data = data[data['label'] == 'spam']
        spam_lengths = spam_data['text'].str.len()
        
        st.write(f"平均長度: {spam_lengths.mean():.0f} 字符")
        st.write(f"中位數長度: {spam_lengths.median():.0f} 字符")
        st.write(f"最長: {spam_lengths.max()} 字符")
        st.write(f"最短: {spam_lengths.min()} 字符")
    
    with col2:
        st.markdown("**正常郵件文本統計**")
        ham_data = data[data['label'] == 'ham']
        ham_lengths = ham_data['text'].str.len()
        
        st.write(f"平均長度: {ham_lengths.mean():.0f} 字符")
        st.write(f"中位數長度: {ham_lengths.median():.0f} 字符")
        st.write(f"最長: {ham_lengths.max()} 字符")
        st.write(f"最短: {ham_lengths.min()} 字符")
    
    # Top Tokens 分析
    st.subheader("🔤 Top Tokens 分析")
    
    # 預處理文本用於分析
    with st.spinner("正在分析文本特徵..."):
        processed_texts = []
        labels = []
        
        for _, row in data.iterrows():
            processed_text = preprocessor.preprocess_text(row['text'])
            processed_texts.append(processed_text)
            labels.append(row['label'])
        
        # 獲取 Top Tokens
        top_tokens = preprocessor.get_top_tokens(processed_texts, labels, n_tokens=20)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**垃圾郵件 Top Tokens**")
        if top_tokens['spam']:
            spam_tokens_fig = visualizer.create_top_tokens_chart(
                top_tokens['spam'], 
                "垃圾郵件高頻詞彙", 
                visualizer.color_palette['spam']
            )
            st.plotly_chart(spam_tokens_fig, use_container_width=True)
        else:
            st.warning("無法獲取垃圾郵件詞彙數據")
    
    with col2:
        st.markdown("**正常郵件 Top Tokens**")
        if top_tokens['ham']:
            ham_tokens_fig = visualizer.create_top_tokens_chart(
                top_tokens['ham'], 
                "正常郵件高頻詞彙", 
                visualizer.color_palette['ham']
            )
            st.plotly_chart(ham_tokens_fig, use_container_width=True)
        else:
            st.warning("無法獲取正常郵件詞彙數據")
    
    # 詞雲圖
    st.subheader("☁️ 詞雲圖")
    
    wordcloud_type = st.radio(
        "選擇詞雲類型",
        ["垃圾郵件", "正常郵件", "全部郵件"],
        horizontal=True,
        help="選擇要生成詞雲的郵件類型"
    )
    
    if wordcloud_type == "垃圾郵件":
        spam_texts = ' '.join([text for text, label in zip(processed_texts, labels) if label == 'spam'])
        if spam_texts.strip():
            wordcloud_fig = visualizer.create_wordcloud(spam_texts, "垃圾郵件詞雲")
            st.pyplot(wordcloud_fig)
        else:
            st.warning("沒有足夠的垃圾郵件數據生成詞雲")
    
    elif wordcloud_type == "正常郵件":
        ham_texts = ' '.join([text for text, label in zip(processed_texts, labels) if label == 'ham'])
        if ham_texts.strip():
            wordcloud_fig = visualizer.create_wordcloud(ham_texts, "正常郵件詞雲")
            st.pyplot(wordcloud_fig)
        else:
            st.warning("沒有足夠的正常郵件數據生成詞雲")
    
    else:  # 全部郵件
        all_texts = ' '.join(processed_texts)
        if all_texts.strip():
            wordcloud_fig = visualizer.create_wordcloud(all_texts, "全部郵件詞雲")
            st.pyplot(wordcloud_fig)
        else:
            st.warning("沒有足夠的數據生成詞雲")
    
    # 數據樣本展示
    st.subheader("📋 數據樣本")
    
    sample_type = st.selectbox(
        "選擇要查看的樣本類型",
        ["全部", "垃圾郵件", "正常郵件"],
        help="選擇要展示的郵件類型"
    )
    
    sample_size = st.slider(
        "樣本數量",
        min_value=5,
        max_value=50,
        value=10,
        help="選擇要顯示的樣本數量"
    )
    
    if sample_type == "全部":
        sample_data = data.sample(n=min(sample_size, len(data)))
    elif sample_type == "垃圾郵件":
        spam_data = data[data['label'] == 'spam']
        sample_data = spam_data.sample(n=min(sample_size, len(spam_data)))
    else:  # 正常郵件
        ham_data = data[data['label'] == 'ham']
        sample_data = ham_data.sample(n=min(sample_size, len(ham_data)))
    
    # 顯示樣本
    for idx, row in sample_data.iterrows():
        with st.expander(f"{row['label'].upper()} - {row['text'][:50]}..."):
            st.write(f"**標籤**: {row['label']}")
            st.write(f"**長度**: {len(row['text'])} 字符")
            st.write(f"**內容**: {row['text']}")
    
    # 數據質量分析
    st.subheader("🔍 數據質量分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**數據完整性**")
        
        # 檢查缺失值
        missing_labels = data['label'].isnull().sum()
        missing_texts = data['text'].isnull().sum()
        
        st.write(f"缺失標籤: {missing_labels}")
        st.write(f"缺失文本: {missing_texts}")
        
        # 檢查空文本
        empty_texts = (data['text'].str.strip() == '').sum()
        st.write(f"空文本: {empty_texts}")
        
        # 檢查重複
        duplicates = data.duplicated().sum()
        st.write(f"重複記錄: {duplicates}")
    
    with col2:
        st.markdown("**標籤分布平衡性**")
        
        spam_ratio = dataset_info.get('spam_percentage', 0) / 100
        ham_ratio = dataset_info.get('ham_percentage', 0) / 100
        
        # 計算不平衡程度
        imbalance_ratio = max(spam_ratio, ham_ratio) / min(spam_ratio, ham_ratio)
        
        st.write(f"垃圾郵件比例: {spam_ratio:.1%}")
        st.write(f"正常郵件比例: {ham_ratio:.1%}")
        st.write(f"不平衡比例: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            st.warning("⚠️ 數據集存在嚴重不平衡")
        elif imbalance_ratio > 2:
            st.info("ℹ️ 數據集存在輕微不平衡")
        else:
            st.success("✅ 數據集相對平衡")
    
    # 導出功能
    st.subheader("💾 數據導出")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("導出數據統計報告", use_container_width=True):
            # 生成統計報告
            report = generate_data_report(data, dataset_info, top_tokens)
            st.download_button(
                label="下載統計報告 (TXT)",
                data=report,
                file_name="spam_data_report.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("導出樣本數據", use_container_width=True):
            # 導出 CSV
            csv_data = data.to_csv(index=False)
            st.download_button(
                label="下載數據集 (CSV)",
                data=csv_data,
                file_name="spam_dataset.csv",
                mime="text/csv"
            )

def generate_data_report(data: pd.DataFrame, dataset_info: dict, top_tokens: dict) -> str:
    """生成數據統計報告"""
    
    report = f"""
垃圾郵件數據集統計報告
===================

數據集基本信息:
- 總郵件數: {dataset_info.get('total_messages', 0):,}
- 垃圾郵件數: {dataset_info.get('spam_count', 0):,} ({dataset_info.get('spam_percentage', 0):.1f}%)
- 正常郵件數: {dataset_info.get('ham_count', 0):,} ({dataset_info.get('ham_percentage', 0):.1f}%)

文本長度統計:
- 平均長度: {dataset_info.get('avg_message_length', 0):.0f} 字符
- 最大長度: {dataset_info.get('max_message_length', 0)} 字符
- 最小長度: {dataset_info.get('min_message_length', 0)} 字符

垃圾郵件 Top 10 詞彙:
"""
    
    for i, (word, count) in enumerate(top_tokens.get('spam', [])[:10]):
        report += f"{i+1:2d}. {word}: {count}\n"
    
    report += "\n正常郵件 Top 10 詞彙:\n"
    
    for i, (word, count) in enumerate(top_tokens.get('ham', [])[:10]):
        report += f"{i+1:2d}. {word}: {count}\n"
    
    report += f"\n報告生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    return report