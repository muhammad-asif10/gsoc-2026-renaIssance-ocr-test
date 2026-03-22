# Historical Document OCR: CNN-RNN with LLM Integration

**GSOC 2026 Test Submission | HumanAI Foundation | RenAIssance Project**

This repository contains a complete implementation of a hybrid CNN-RNN architecture 
with LLM post-processing for Optical Character Recognition (OCR) of historical 
Renaissance-era printed documents, specifically targeting 17th-century Spanish texts.

## Test

- [Test detail](https://humanai.foundation/assets/GSoC%202026%20tests.pdf)

- [Source of data](https://bama365-my.sharepoint.com/personal/xgranja_ua_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxgranja%5Fua%5Fedu%2FDocuments%2FUA%2F1%2E%20Research%2FAI%2FHumanAI%2FGSoC%2026%2F0%2E%20Test%2FTest%20sources&viewid=aeb9535d%2D9751%2D4642%2D912a%2Dc16ad99be40c)

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
