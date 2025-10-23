# Project Context

## Purpose
電子郵件垃圾分類系統 - 提供智能的垃圾郵件檢測和分類功能

## Tech Stack
- Python 3.8+
- 機器學習框架 (scikit-learn, TensorFlow/PyTorch)
- Streamlit (Web 應用界面)
- SQLite/PostgreSQL (數據存儲)
- Pickle/Joblib (模型持久化)
- Docker (容器化部署)

## Project Conventions

### Code Style
- 使用 Black 進行代碼格式化
- 遵循 PEP 8 標準
- 使用 Type Hints
- 函數和變量使用 snake_case
- 類名使用 PascalCase

### Architecture Patterns
- 分層架構：UI Layer (Streamlit) → Service Layer → Data Layer
- 模塊化設計：分離特徵提取、模型預測和數據處理
- 單體應用架構（適合 Streamlit 部署）
- 狀態管理使用 Streamlit Session State

### Testing Strategy
- 單元測試覆蓋率 > 80%
- 使用 pytest 框架
- Streamlit 應用測試
- 性能測試用於 ML 模型

### Git Workflow
- 使用 Git Flow 分支策略
- 提交信息格式：[Phase-X] type: description
- 代碼審查必須通過才能合併

## Domain Context
- 垃圾郵件檢測需要處理多語言內容
- 需要支持實時和批量處理模式
- 模型需要定期重訓練以適應新的垃圾郵件模式
- 隱私保護：不存儲郵件內容，只保留特徵和分類結果

## Important Constraints
- 響應時間 < 500ms (實時檢測)
- 支持每日處理 100萬封郵件
- 準確率 > 95%
- 符合 GDPR 數據保護要求

## External Dependencies
- SMTP 服務器集成
- 第三方威脅情報 API
- 雲存儲服務 (模型文件)
- 監控和日誌服務
