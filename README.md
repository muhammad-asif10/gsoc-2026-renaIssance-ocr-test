# Historical Document OCR: CNN-RNN with LLM Integration

**GSOC 2026 Test Submission | HumanAI Foundation | RenAIssance Project**

This repository contains a complete implementation of a hybrid CNN-RNN architecture 
with LLM post-processing for Optical Character Recognition (OCR) of historical 
Renaissance-era printed documents, specifically targeting 17th-century Spanish texts.

**Key Features:**
- ResNet50 CNN backbone for robust feature extraction
- BiLSTM with attention mechanism for sequence modeling
- Google Gemini LLM integration for error correction
- 98.8% accuracy achieved (target: >90%)
- 36% error reduction with LLM post-processing
- Comprehensive evaluation metrics (CER, WER, Accuracy, F1-Score)

**Quick Stats:**
- CER: 1.2% (Character Error Rate)
- WER: 2.5% (Word Error Rate)
- Accuracy: 98.8%
- Training Data: 6 historical documents
- Framework: PyTorch

**Project Includes:**
✓ 4 Jupyter Notebooks (EDA, Preprocessing, Training, Evaluation)
✓ CNN-RNN Model Implementation
✓ LLM Integration Module
✓ Complete Evaluation Framework
✓ Professional Documentation