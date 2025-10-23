"""
配置文件
統一管理應用配置和字體設置
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import warnings

# 忽略字體警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def setup_matplotlib_config():
    """設置 matplotlib 配置"""
    try:
        # 設置字體
        system = platform.system()
        if system == "Windows":
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans']
        elif system == "Darwin":  # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Arial', 'DejaVu Sans']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Arial', 'DejaVu Sans']
        
        # 解決負號顯示問題
        plt.rcParams['axes.unicode_minus'] = False
        
        # 設置圖表樣式
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        
    except Exception:
        # 如果設置失敗，使用默認配置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

# 應用配置
APP_CONFIG = {
    'title': 'Spam Email Classifier - Phase 1',
    'icon': '📧',
    'layout': 'wide',
    'colors': {
        'spam': '#FF6B6B',
        'ham': '#4ECDC4',
        'primary': '#FF6B6B',
        'secondary': '#4ECDC4'
    },
    'model_names': ['Naive Bayes', 'SVM', 'Random Forest', 'Logistic Regression']
}

# 文本標籤映射（避免中文字體問題）
LABELS = {
    'zh': {
        'spam': '垃圾郵件',
        'ham': '正常郵件',
        'classification': '分類',
        'confidence': '置信度',
        'accuracy': '準確率',
        'precision': '精確率',
        'recall': '召回率'
    },
    'en': {
        'spam': 'Spam',
        'ham': 'Ham',
        'classification': 'Classification',
        'confidence': 'Confidence',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall'
    }
}

# 初始化配置
setup_matplotlib_config()