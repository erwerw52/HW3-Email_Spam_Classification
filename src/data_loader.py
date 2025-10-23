"""
數據載入器模塊
負責載入和管理垃圾郵件數據集
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import streamlit as st
import os

class DataLoader:
    """數據載入器類"""
    
    def __init__(self):
        self.raw_data_path = "dataset/sms_spam_no_header.csv"
        self.processed_data_path = "dataset/processed/sms_spam_clean.csv"
        
    @st.cache_data
    def load_raw_dataset(_self) -> pd.DataFrame:
        """載入原始垃圾郵件數據集"""
        try:
            # 載入原始數據集
            df = pd.read_csv(_self.raw_data_path, header=None, names=['label', 'text'])
            
            # 基本數據清理
            df['label'] = df['label'].str.strip().str.lower()
            df['text'] = df['text'].str.strip()
            
            # 移除空值
            df = df.dropna()
            
            return df
        except FileNotFoundError:
            st.error(f"找不到數據文件: {_self.raw_data_path}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"載入數據時發生錯誤: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_processed_dataset(_self) -> pd.DataFrame:
        """載入處理過的數據集"""
        try:
            if os.path.exists(_self.processed_data_path):
                df = pd.read_csv(_self.processed_data_path)
                return df
            else:
                # 如果沒有處理過的數據，返回原始數據
                return _self.load_raw_dataset()
        except Exception as e:
            st.error(f"載入處理過的數據時發生錯誤: {str(e)}")
            return _self.load_raw_dataset()
    
    def get_dataset_info(self) -> Dict:
        """獲取數據集基本信息"""
        df = self.load_raw_dataset()
        
        if df.empty:
            return {}
        
        info = {
            'total_messages': len(df),
            'spam_count': len(df[df['label'] == 'spam']),
            'ham_count': len(df[df['label'] == 'ham']),
            'spam_percentage': len(df[df['label'] == 'spam']) / len(df) * 100,
            'ham_percentage': len(df[df['label'] == 'ham']) / len(df) * 100,
            'avg_message_length': df['text'].str.len().mean(),
            'max_message_length': df['text'].str.len().max(),
            'min_message_length': df['text'].str.len().min()
        }
        
        return info
    
    def get_sample_emails(self) -> Dict[str, List[str]]:
        """獲取範例郵件"""
        df = self.load_raw_dataset()
        
        if df.empty:
            return {'spam': [], 'ham': []}
        
        # 獲取範例
        spam_samples = df[df['label'] == 'spam']['text'].head(3).tolist()
        ham_samples = df[df['label'] == 'ham']['text'].head(3).tolist()
        
        return {
            'spam': spam_samples,
            'ham': ham_samples
        }
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分割數據為訓練集和測試集"""
        from sklearn.model_selection import train_test_split
        
        df = self.load_raw_dataset()
        
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # 分割數據
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df['label']
        )
        
        return train_df, test_df