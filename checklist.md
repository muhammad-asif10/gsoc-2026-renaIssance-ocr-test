DATA PREPARATION (Phase 1)
☐ PDF to image conversion (PyMuPDF)
☐ Line segmentation with deskewing
☐ Historical document preprocessing
☐ Metadata creation

ANNOTATION (Phase 2)
☐ Transcription GUI
☐ Manual transcription of 35-50 pages
☐ Semi-automatic Tesseract correction
☐ Save ground truth

AUGMENTATION (Phase 3)
☐ Historical document augmentation
☐ Create 3x versions per image
☐ Handle aging effects, ink variations

MODEL (Phase 4)
☐ Weighted CNN-RNN architecture
☐ Special handling for diacritics
☐ Weighted loss function
☐ Training pipeline

DECODING (Phase 5)
☐ Constrained beam search
☐ Spanish lexicon integration
☐ N-gram language model

LLM (Phase 6)
☐ Gemini API integration
☐ Post-processing pipeline
☐ Context-aware correction

EVALUATION (Phase 7)
☐ CER, WER, similarity metrics
☐ Accuracy benchmarking
☐ Error analysis