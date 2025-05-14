# EduQA

EduQA 是一個基於 RAG（檢索增強生成）的教育問答系統，支援多語言嵌入與自動化問答生成。

## 特色
- 支援多種語言模型（如 OpenAI、HuggingFace、Ollama）
- 整合 Chroma 向量資料庫，實現高效知識檢索
- 自動化處理 CSV 問答資料，並可檢查簡體中文
- 可擴充的問答處理流程（Pipeline）

## 專案結構
- `Agent/`：主要程式碼
  - `ReAct.py`：主流程與 RAG pipeline
  - `LLM_prompt.py`：提示詞模板
  - `SimplifiedChineseChecker.py`：簡體中文檢查工具
  - `utils.py`：輔助工具
- `user_manual_files/`：教育相關 CSV 資料集
- `QA_v1.0.csv`：輸入問答資料
- `output.csv`：輸出結果

## 安裝需求
- Python 3.12.2
- 主要套件：
  - langchain
  - chromadb
  - huggingface_hub
  - openai
  - 其他請參考程式碼 import

## 使用方式
1. 準備好問答 CSV 檔（需包含 Question, Answer, URL 欄位），如 QA_v1.0.csv。
2. 執行 `Agent/ReAct.py`，自動處理問答資料並產生 output.csv。
3. 可自訂向量資料庫與模型參數。

## 主要功能
- 問答資料自動嵌入與檢索
- 多步驟問答生成流程
- 結果自動寫入 CSV
- 支援簡體中文檢查