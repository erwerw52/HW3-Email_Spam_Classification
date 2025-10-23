"""
多模型分類器模塊
支持多種機器學習算法進行垃圾郵件分類
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import streamlit as st
import joblib
import os
from datetime import datetime

# 機器學習模型
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 評估指標
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    classification_report
)
from sklearn.model_selection import cross_val_score

# 導入自定義模塊
from .preprocessing import TextPreprocessor
from .data_loader import DataLoader

class MultiModelClassifier:
    """多模型分類器類"""
    
    def __init__(self, fast_mode: bool = True):
        """
        初始化多模型分類器
        
        Args:
            fast_mode: 是否使用快速模式（優化訓練速度）
        """
        if fast_mode:
            # 快速模式：優化速度
            self.models = {
                'Naive Bayes': MultinomialNB(),
                'SVM': SVC(
                    kernel='linear',  # 線性核函數最快
                    probability=True, 
                    random_state=42,
                    C=0.1,  # 較小的 C 值加快訓練
                    max_iter=500
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=30,  # 減少樹的數量
                    random_state=42,
                    n_jobs=-1,
                    max_depth=8,
                    min_samples_split=10
                ),
                'Logistic Regression': LogisticRegression(
                    random_state=42, 
                    max_iter=500,
                    solver='liblinear',
                    C=0.1
                )
            }
        else:
            # 標準模式：平衡速度和性能
            self.models = {
                'Naive Bayes': MultinomialNB(),
                'SVM': SVC(
                    kernel='linear',
                    probability=True, 
                    random_state=42,
                    C=1.0,
                    max_iter=1000
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=15
                ),
                'Logistic Regression': LogisticRegression(
                    random_state=42, 
                    max_iter=1000,
                    solver='liblinear',
                    C=1.0
                )
            }
        
        self.trained_models = {}
        self.preprocessor = TextPreprocessor()
        self.data_loader = DataLoader()
        self.model_performance = {}
        self.is_trained = False
        self.fast_mode = fast_mode
    
    def get_trained_models(self) -> List[str]:
        """獲取已訓練的模型列表"""
        return list(self.trained_models.keys())
    
    def get_model_status(self) -> Dict[str, bool]:
        """獲取所有模型的訓練狀態"""
        status = {}
        for model_name in self.models.keys():
            status[model_name] = model_name in self.trained_models
        return status
        
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """訓練所有模型"""
        self.trained_models = {}
        self.model_performance = {}
        
        # 創建進度條
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_models = len(self.models)
        
        for i, (name, model) in enumerate(self.models.items()):
            try:
                status_text.text(f'正在訓練 {name} 模型... ({i+1}/{total_models})')
                
                # 訓練模型
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                
                # 預測
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # 計算性能指標
                performance = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, pos_label='spam'),
                    'recall': recall_score(y_test, y_pred, pos_label='spam'),
                    'f1': f1_score(y_test, y_pred, pos_label='spam'),
                    'auc': roc_auc_score(y_test == 'spam', y_pred_proba),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'y_test': y_test
                }
                
                # 簡化交叉驗證（減少 CV 折數以加快速度）
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                performance['cv_mean'] = cv_scores.mean()
                performance['cv_std'] = cv_scores.std()
                
                self.model_performance[name] = performance
                
                # 更新進度
                progress_bar.progress((i + 1) / total_models)
                st.success(f"✅ {name} 訓練完成 - 準確率: {performance['accuracy']:.3f}")
                

                
            except Exception as e:
                st.error(f"❌ {name} 訓練失敗: {str(e)}")
                # 記錄失敗的模型，但繼續訓練其他模型
                st.warning(f"跳過 {name} 模型，繼續訓練其他模型...")
                continue
        
        # 清理進度顯示
        progress_bar.empty()
        status_text.empty()
        
        # 檢查訓練結果
        total_models = len(self.models)
        successful_models = len(self.trained_models)
        
        if successful_models > 0:
            self.is_trained = True
            st.success(f"🎉 模型訓練完成！成功訓練了 {successful_models}/{total_models} 個模型")
            
            if successful_models < total_models:
                failed_models = set(self.models.keys()) - set(self.trained_models.keys())
                st.warning(f"⚠️ 以下模型訓練失敗: {', '.join(failed_models)}")
        else:
            st.error("❌ 所有模型訓練失敗！請檢查數據和配置。")
            self.is_trained = False
    
    def predict(self, text: str, model_name: str = 'Naive Bayes') -> Dict[str, Any]:
        """預測單個文本"""
        # 詳細的錯誤檢查
        if not self.is_trained:
            return {
                'is_spam': False,
                'confidence': 0.0,
                'spam_probability': 0.0,
                'processing_time': 0.0,
                'error': '模型尚未訓練，請先訓練模型'
            }
        
        if model_name not in self.trained_models:
            available_models = list(self.trained_models.keys())
            return {
                'is_spam': False,
                'confidence': 0.0,
                'spam_probability': 0.0,
                'processing_time': 0.0,
                'error': f'模型 "{model_name}" 不存在。可用模型: {available_models}'
            }
        
        start_time = datetime.now()
        
        try:
            # 預處理文本
            processed_text = self.preprocessor.preprocess_text(text)
            
            # 特徵提取
            features = self.preprocessor.extract_tfidf_features([processed_text])
            
            # 預測
            model = self.trained_models[model_name]
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # 計算處理時間
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # 獲取垃圾郵件機率
            spam_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            
            return {
                'is_spam': prediction == 'spam',
                'confidence': max(probabilities),
                'spam_probability': spam_prob,
                'processing_time': processing_time,
                'processed_text': processed_text,
                'model_used': model_name
            }
            
        except Exception as e:
            return {
                'is_spam': False,
                'confidence': 0.0,
                'spam_probability': 0.0,
                'processing_time': 0.0,
                'error': str(e)
            }
    
    def get_model_comparison(self) -> pd.DataFrame:
        """獲取模型比較表格"""
        if not self.model_performance:
            return pd.DataFrame()
        
        comparison_data = []
        for name, perf in self.model_performance.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{perf['accuracy']:.4f}",
                'Precision': f"{perf['precision']:.4f}",
                'Recall': f"{perf['recall']:.4f}",
                'F1-Score': f"{perf['f1']:.4f}",
                'AUC': f"{perf['auc']:.4f}",
                'CV Mean': f"{perf['cv_mean']:.4f}",
                'CV Std': f"{perf['cv_std']:.4f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_roc_data(self) -> Dict[str, Dict]:
        """獲取 ROC 曲線數據"""
        roc_data = {}
        
        for name, perf in self.model_performance.items():
            y_test_binary = (perf['y_test'] == 'spam').astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, perf['y_pred_proba'])
            
            roc_data[name] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': perf['auc']
            }
        
        return roc_data
    
    def get_pr_data(self) -> Dict[str, Dict]:
        """獲取 Precision-Recall 曲線數據"""
        pr_data = {}
        
        for name, perf in self.model_performance.items():
            y_test_binary = (perf['y_test'] == 'spam').astype(int)
            precision, recall, _ = precision_recall_curve(y_test_binary, perf['y_pred_proba'])
            
            pr_data[name] = {
                'precision': precision,
                'recall': recall
            }
        
        return pr_data
    
    def get_confusion_matrix_data(self, model_name: str) -> np.ndarray:
        """獲取混淆矩陣數據"""
        if model_name not in self.model_performance:
            return np.array([])
        
        perf = self.model_performance[model_name]
        return confusion_matrix(perf['y_test'], perf['y_pred'])
    
    def save_models(self, filepath: str = "models/trained_models.pkl"):
        """保存訓練好的模型"""
        if not self.is_trained:
            st.warning("沒有訓練好的模型可以保存")
            return False
        
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存模型和預處理器
            model_data = {
                'trained_models': self.trained_models,
                'preprocessor': self.preprocessor,
                'model_performance': self.model_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            st.success(f"模型已保存到 {filepath}")
            return True
            
        except Exception as e:
            st.error(f"保存模型時發生錯誤: {str(e)}")
            return False
    
    def load_models(self, filepath: str = "models/trained_models.pkl"):
        """載入訓練好的模型"""
        try:
            if not os.path.exists(filepath):
                st.warning(f"模型文件不存在: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.trained_models = model_data['trained_models']
            self.preprocessor = model_data['preprocessor']
            self.model_performance = model_data['model_performance']
            self.is_trained = True
            
            st.success(f"模型已從 {filepath} 載入")
            return True
            
        except Exception as e:
            st.error(f"載入模型時發生錯誤: {str(e)}")
            return False
    
    def get_feature_importance(self, model_name: str, n_features: int = 20) -> List[Tuple[str, float]]:
        """獲取特徵重要性"""
        if model_name not in self.trained_models:
            return []
        
        model = self.trained_models[model_name]
        feature_names = self.preprocessor.get_feature_names()
        
        if hasattr(model, 'feature_importances_'):
            # Random Forest
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Logistic Regression, SVM
            importances = np.abs(model.coef_[0])
        else:
            # Naive Bayes
            importances = np.abs(model.feature_log_prob_[1] - model.feature_log_prob_[0])
        
        # 獲取前 n 個重要特徵
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:n_features]