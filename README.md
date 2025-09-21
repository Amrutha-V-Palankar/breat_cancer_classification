# Real-Time Breast Cancer Detection

## Overview
A real-time deep learning app for IDC detection using a trained DenseNet model.

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run streamlit_app.py
   ```

## Notes
- Ensure `model/densenet121_idc.h5` exists (trained model).
- Upload histopathology image to get prediction.
