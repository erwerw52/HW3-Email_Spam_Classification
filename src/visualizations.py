"""
數據可視化模塊
提供各種圖表和可視化功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
import warnings

# 導入配置
from .config import APP_CONFIG, LABELS

# 忽略字體警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

class Visualizer:
    """可視化器類"""
    
    def __init__(self):
        self.color_palette = APP_CONFIG['colors']
        self.labels = LABELS['en']  # 使用英文標籤避免字體問題
    
    def create_class_distribution_chart(self, data: pd.DataFrame) -> go.Figure:
        """創建類別分布圖表"""
        class_counts = data['label'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=class_counts.index,
                y=class_counts.values,
                marker_color=[self.color_palette['ham'], self.color_palette['spam']],
                text=class_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Email Class Distribution",
            xaxis_title="Class",
            yaxis_title="Count",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_top_tokens_chart(self, tokens_data: List[Tuple[str, int]], 
                               title: str, color: str) -> go.Figure:
        """創建 Top Tokens 橫向條形圖"""
        if not tokens_data:
            return go.Figure()
        
        words, counts = zip(*tokens_data)
        
        fig = go.Figure(data=[
            go.Bar(
                y=words[::-1],  # 反轉順序，最高的在上面
                x=counts[::-1],
                orientation='h',
                marker_color=color,
                text=counts[::-1],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Frequency",
            yaxis_title="Tokens",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_roc_curves(self, roc_data: Dict[str, Dict]) -> go.Figure:
        """創建 ROC 曲線圖"""
        fig = go.Figure()
        
        # 添加對角線（隨機分類器）
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier',
            showlegend=True
        ))
        
        # 為每個模型添加 ROC 曲線
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, (model_name, data) in enumerate(roc_data.items()):
            fig.add_trace(go.Scatter(
                x=data['fpr'],
                y=data['tpr'],
                mode='lines',
                name=f"{model_name} (AUC = {data['auc']:.3f})",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=500,
            height=400,
            legend=dict(x=0.6, y=0.1)
        )
        
        return fig
    
    def create_precision_recall_curves(self, pr_data: Dict[str, Dict]) -> go.Figure:
        """創建 Precision-Recall 曲線圖"""
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, (model_name, data) in enumerate(pr_data.items()):
            fig.add_trace(go.Scatter(
                x=data['recall'],
                y=data['precision'],
                mode='lines',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title="Precision-Recall Curves",
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=500,
            height=400,
            legend=dict(x=0.1, y=0.1)
        )
        
        return fig
    
    def create_confusion_matrix(self, cm_data: np.ndarray, 
                               labels: List[str] = ['ham', 'spam']) -> go.Figure:
        """創建混淆矩陣熱力圖"""
        fig = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm_data,
            texttemplate="%{text}",
            textfont={"size": 16},
            showscale=True
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=400,
            height=400
        )
        
        return fig
    
    def create_spam_probability_bar(self, probability: float) -> go.Figure:
        """創建垃圾郵件機率條形圖"""
        # 決定顏色
        color = self.color_palette['spam'] if probability > 0.5 else self.color_palette['ham']
        
        fig = go.Figure(go.Bar(
            x=[probability],
            y=['Spam Probability'],
            orientation='h',
            marker_color=color,
            text=f"{probability:.1%}",
            textposition='auto',
        ))
        
        fig.update_layout(
            xaxis=dict(range=[0, 1], tickformat='.0%'),
            yaxis=dict(showticklabels=False),
            height=100,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        # 添加閾值線
        fig.add_vline(x=0.5, line_dash="dash", line_color="black", opacity=0.5)
        
        return fig
    
    def create_wordcloud(self, text_data: str, title: str) -> plt.Figure:
        """創建詞雲圖"""
        if not text_data.strip():
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'No sufficient data for wordcloud', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # 生成詞雲
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis',
            prefer_horizontal=0.9,
            min_font_size=10
        ).generate(text_data)
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        # 使用英文標題避免字體問題
        ax.set_title(title.replace('垃圾郵件', 'Spam').replace('正常郵件', 'Ham'), 
                    fontsize=16, pad=20)
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance: List[Tuple[str, float]], 
                                      title: str) -> go.Figure:
        """創建特徵重要性圖表"""
        if not feature_importance:
            return go.Figure()
        
        features, importances = zip(*feature_importance)
        
        fig = go.Figure(data=[
            go.Bar(
                y=features[::-1],  # 反轉順序
                x=importances[::-1],
                orientation='h',
                marker_color=self.color_palette['primary'],
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Features",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_metrics_cards(self, metrics: Dict[str, float]) -> None:
        """創建指標卡片"""
        cols = st.columns(len(metrics))
        
        for i, (metric_name, value) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(value, float):
                    if metric_name.lower() in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                        st.metric(metric_name, f"{value:.1%}")
                    else:
                        st.metric(metric_name, f"{value:.2f}")
                else:
                    st.metric(metric_name, str(value))
    
    def create_text_length_distribution(self, data: pd.DataFrame) -> go.Figure:
        """創建文本長度分布圖"""
        # 計算文本長度
        data['text_length'] = data['text'].str.len()
        
        fig = go.Figure()
        
        # 為每個類別創建直方圖
        for label in data['label'].unique():
            subset = data[data['label'] == label]
            fig.add_trace(go.Histogram(
                x=subset['text_length'],
                name=label,
                opacity=0.7,
                marker_color=self.color_palette[label]
            ))
        
        fig.update_layout(
            title="Text Length Distribution",
            xaxis_title="Text Length (Characters)",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        
        return fig