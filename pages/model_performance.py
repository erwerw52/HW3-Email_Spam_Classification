"""
Model Performance 頁面
模型性能分析和比較
"""

import streamlit as st
import pandas as pd
from typing import Any

def show_page(classifier: Any, visualizer: Any, model_trained: bool):
    """顯示 Model Performance 頁面"""
    
    st.title("📊 Model Performance")
    st.markdown("分析和比較不同機器學習模型的性能表現")
    
    if not model_trained:
        st.warning("⚠️ 請先在側邊欄訓練模型才能查看性能分析")
        return
    
    # 模型比較表格
    st.subheader("🏆 模型性能比較")
    
    comparison_df = classifier.get_model_comparison()
    if not comparison_df.empty:
        st.dataframe(comparison_df, use_container_width=True)
        
        # 最佳模型推薦
        best_model_idx = comparison_df['AUC'].astype(float).idxmax()
        best_model = comparison_df.iloc[best_model_idx]['Model']
        best_auc = comparison_df.iloc[best_model_idx]['AUC']
        
        st.success(f"🥇 **推薦模型**: {best_model} (AUC: {best_auc})")
    else:
        st.error("無法獲取模型性能數據")
        return
    
    # ROC 和 PR 曲線
    st.subheader("📈 性能曲線分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**ROC 曲線**")
        roc_data = classifier.get_roc_data()
        if roc_data:
            roc_fig = visualizer.create_roc_curves(roc_data)
            st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.error("無法生成 ROC 曲線")
    
    with col2:
        st.markdown("**Precision-Recall 曲線**")
        pr_data = classifier.get_pr_data()
        if pr_data:
            pr_fig = visualizer.create_precision_recall_curves(pr_data)
            st.plotly_chart(pr_fig, use_container_width=True)
        else:
            st.error("無法生成 Precision-Recall 曲線")
    
    # 混淆矩陣
    st.subheader("🎯 混淆矩陣分析")
    
    # 模型選擇
    model_names = list(classifier.trained_models.keys())
    selected_model = st.selectbox(
        "選擇要分析的模型",
        model_names,
        help="選擇一個模型來查看其混淆矩陣"
    )
    
    if selected_model:
        cm_data = classifier.get_confusion_matrix_data(selected_model)
        if cm_data.size > 0:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                cm_fig = visualizer.create_confusion_matrix(cm_data)
                st.plotly_chart(cm_fig, use_container_width=True)
            
            with col2:
                st.markdown("**混淆矩陣解讀**")
                
                tn, fp, fn, tp = cm_data.ravel()
                
                st.write(f"**True Negatives (TN)**: {tn}")
                st.write(f"**False Positives (FP)**: {fp}")
                st.write(f"**False Negatives (FN)**: {fn}")
                st.write(f"**True Positives (TP)**: {tp}")
                
                # 計算額外指標
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                st.write(f"**敏感度 (Sensitivity)**: {sensitivity:.3f}")
                st.write(f"**特異度 (Specificity)**: {specificity:.3f}")
                
                # 錯誤分析
                st.markdown("**錯誤分析**")
                if fp > 0:
                    st.write(f"• {fp} 個正常郵件被誤判為垃圾郵件")
                if fn > 0:
                    st.write(f"• {fn} 個垃圾郵件被誤判為正常郵件")
    
    # 特徵重要性分析
    st.subheader("🔍 特徵重要性分析")
    
    feature_model = st.selectbox(
        "選擇模型查看特徵重要性",
        model_names,
        key="feature_model",
        help="不同模型的特徵重要性計算方式不同"
    )
    
    if feature_model:
        n_features = st.slider(
            "顯示特徵數量",
            min_value=10,
            max_value=50,
            value=20,
            help="選擇要顯示的重要特徵數量"
        )
        
        feature_importance = classifier.get_feature_importance(feature_model, n_features)
        
        if feature_importance:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                importance_fig = visualizer.create_feature_importance_chart(
                    feature_importance, 
                    f"{feature_model} - Top {n_features} 重要特徵"
                )
                st.plotly_chart(importance_fig, use_container_width=True)
            
            with col2:
                st.markdown("**特徵重要性表格**")
                importance_df = pd.DataFrame(
                    feature_importance, 
                    columns=['特徵', '重要性']
                )
                importance_df['重要性'] = importance_df['重要性'].round(4)
                st.dataframe(importance_df, use_container_width=True)
        else:
            st.warning("無法獲取特徵重要性數據")
    
    # 模型詳細信息
    st.subheader("ℹ️ 模型詳細信息")
    
    with st.expander("查看模型配置和參數"):
        for model_name, model in classifier.trained_models.items():
            st.markdown(f"**{model_name}**")
            st.write(f"模型類型: {type(model).__name__}")
            
            # 顯示模型參數
            params = model.get_params()
            important_params = {}
            
            # 只顲示重要參數
            if model_name == 'SVM':
                important_params = {k: v for k, v in params.items() 
                                 if k in ['C', 'kernel', 'gamma']}
            elif model_name == 'Random Forest':
                important_params = {k: v for k, v in params.items() 
                                 if k in ['n_estimators', 'max_depth', 'min_samples_split']}
            elif model_name == 'Logistic Regression':
                important_params = {k: v for k, v in params.items() 
                                 if k in ['C', 'penalty', 'solver']}
            else:
                important_params = {k: v for k, v in params.items() 
                                 if k in ['alpha', 'fit_prior']}
            
            for param, value in important_params.items():
                st.write(f"  • {param}: {value}")
            
            st.markdown("---")
    
    # 性能總結
    st.subheader("📋 性能總結")
    
    if not comparison_df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**最高準確率**")
            best_acc_idx = comparison_df['Accuracy'].astype(float).idxmax()
            best_acc_model = comparison_df.iloc[best_acc_idx]['Model']
            best_acc_value = comparison_df.iloc[best_acc_idx]['Accuracy']
            st.metric(best_acc_model, best_acc_value)
        
        with col2:
            st.markdown("**最高 F1-Score**")
            best_f1_idx = comparison_df['F1-Score'].astype(float).idxmax()
            best_f1_model = comparison_df.iloc[best_f1_idx]['Model']
            best_f1_value = comparison_df.iloc[best_f1_idx]['F1-Score']
            st.metric(best_f1_model, best_f1_value)
        
        with col3:
            st.markdown("**最高 AUC**")
            best_auc_idx = comparison_df['AUC'].astype(float).idxmax()
            best_auc_model = comparison_df.iloc[best_auc_idx]['Model']
            best_auc_value = comparison_df.iloc[best_auc_idx]['AUC']
            st.metric(best_auc_model, best_auc_value)
        
        # 建議
        st.markdown("### 💡 模型選擇建議")
        
        avg_auc = comparison_df['AUC'].astype(float).mean()
        
        if float(best_auc_value) > 0.95:
            st.success("🎉 所有模型表現優秀！推薦使用 AUC 最高的模型。")
        elif float(best_auc_value) > 0.90:
            st.info("👍 模型表現良好。可以考慮進一步調參優化。")
        else:
            st.warning("⚠️ 模型表現有待改善。建議檢查數據質量或嘗試其他特徵工程方法。")