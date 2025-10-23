"""
文本預處理模塊
負責文本清理、標準化和特徵提取
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import nltk

# 嘗試導入 NLTK 功能，如果失敗則使用備用方案
try:
    import nltk
    # 嘗試下載必要的 NLTK 數據
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            pass
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False
    # 備用停用詞列表
    ENGLISH_STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once'
    }

class TextPreprocessor:
    """文本預處理器類"""
    
    def __init__(self):
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = ENGLISH_STOPWORDS
        else:
            self.stop_words = ENGLISH_STOPWORDS
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        
    def clean_text(self, text: str) -> str:
        """基本文本清理"""
        if not isinstance(text, str):
            return ""
        
        # 轉換為小寫
        text = text.lower()
        
        # 移除 HTML 標籤
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除 URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 移除電子郵件地址
        text = re.sub(r'\S+@\S+', '', text)
        
        # 移除電話號碼
        text = re.sub(r'\b\d{10,}\b', '<PHONE>', text)
        
        # 替換數字為 <NUM>
        text = re.sub(r'\b\d+\b', '<NUM>', text)
        
        # 移除標點符號
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 移除多餘空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """移除停用詞"""
        if not isinstance(text, str):
            return ""
        
        # 使用 NLTK 或備用的簡單分詞
        if NLTK_AVAILABLE:
            try:
                words = word_tokenize(text)
            except:
                # 如果 NLTK 分詞失敗，使用簡單分詞
                words = text.split()
        else:
            # 備用簡單分詞
            words = text.split()
        
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True) -> str:
        """完整的文本預處理流程"""
        # 基本清理
        cleaned_text = self.clean_text(text)
        
        # 移除停用詞（可選）
        if remove_stopwords:
            cleaned_text = self.remove_stopwords(cleaned_text)
        
        return cleaned_text
    
    def fit_vectorizers(self, texts: List[str], max_features: int = 5000):
        """訓練向量化器"""
        # TF-IDF 向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # 1-gram 和 2-gram
            min_df=2,
            max_df=0.95
        )
        
        # 詞頻向量化器
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # 訓練向量化器
        self.tfidf_vectorizer.fit(texts)
        self.count_vectorizer.fit(texts)
    
    def extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """提取 TF-IDF 特徵"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF 向量化器尚未訓練，請先調用 fit_vectorizers()")
        
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def extract_count_features(self, texts: List[str]) -> np.ndarray:
        """提取詞頻特徵"""
        if self.count_vectorizer is None:
            raise ValueError("詞頻向量化器尚未訓練，請先調用 fit_vectorizers()")
        
        return self.count_vectorizer.transform(texts).toarray()
    
    def get_top_tokens(self, texts: List[str], labels: List[str], n_tokens: int = 20) -> Dict[str, List[Tuple[str, int]]]:
        """獲取每個類別的高頻詞彙"""
        spam_texts = [text for text, label in zip(texts, labels) if label == 'spam']
        ham_texts = [text for text, label in zip(texts, labels) if label == 'ham']
        
        # 合併所有文本並計算詞頻
        spam_words = ' '.join(spam_texts).split()
        ham_words = ' '.join(ham_texts).split()
        
        spam_counter = Counter(spam_words)
        ham_counter = Counter(ham_words)
        
        return {
            'spam': spam_counter.most_common(n_tokens),
            'ham': ham_counter.most_common(n_tokens)
        }
    
    def get_feature_names(self) -> List[str]:
        """獲取特徵名稱"""
        if self.tfidf_vectorizer is None:
            return []
        
        return self.tfidf_vectorizer.get_feature_names_out().tolist()
    
    def calculate_text_stats(self, texts: List[str]) -> Dict[str, float]:
        """計算文本統計信息"""
        if not texts:
            return {}
        
        lengths = [len(text.split()) for text in texts]
        char_lengths = [len(text) for text in texts]
        
        return {
            'avg_word_count': np.mean(lengths),
            'max_word_count': np.max(lengths),
            'min_word_count': np.min(lengths),
            'avg_char_count': np.mean(char_lengths),
            'max_char_count': np.max(char_lengths),
            'min_char_count': np.min(char_lengths)
        }