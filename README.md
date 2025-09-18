
# Real‑Time Visual Speech Recognition (LipBuddy) — Streamlit Demo (TensorFlow/Keras)



## Repository layout 

```
├── app
│   ├── animation.gif
│   ├── modelutil.py
│   ├── streamlitapp.py
│   ├── test_video.mp4
│   └── utils.py
└── LipNet.ipynb
```
The runnable app lives entirely under **`app/`**. The notebook is illustrative. 

---

## Environment

**Python 3.9+** (CPU or GPU)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install streamlit tensorflow opencv-python imageio
```

**System dependency (required):** **ffmpeg**  
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`  
- macOS (Homebrew): `brew install ffmpeg`  
- Windows: Install ffmpeg and ensure it’s on **PATH**. 

---

## Data & weights

**Dataset layout (GRID‑style expected at repo root):**
```
data/
├── s1/                 # video clips (.mpg)
│   ├── bbaf2n.mpg
│   └── ...
└── alignments/
    └── s1/
        ├── bbaf2n.align
        └── ...
```

> A small `app/test_video.mp4` is included for preview, but full decoding expects the GRID‑style folders above.

---

## Run the app

From the repository root:
```bash
streamlit run app/streamlitapp.py
```
In the UI:
1) Select a clip from `data/s1` in the sidebar.  
2) The left pane shows the **converted** `.mpg → .mp4` preview (via `ffmpeg`).  
3) The right pane shows the **model view** (mouth ROI animation) and **decoded text** produced by CTC. 

---

## How it works (end‑to‑end)

- **Load & convert**: the UI shells out to `ffmpeg` to make an `.mp4` preview for Streamlit.  
- **Frame processing**: `utils.load_data()` uses OpenCV to load frames, converts to grayscale, and crops a **mouth ROI**:
  - Crop slice: `frame[190:236, 80:220, :]` → **46×140×1** per frame
  - Normalize per clip: z‑score to `tf.float32`
- **Sequence length**: **75 frames**, stacked as `(75, 46, 140, 1)`.
- **Model**: 3D Conv blocks → BiLSTM×2 → Dense softmax; **CTC decode** with greedy `tf.keras.backend.ctc_decode` and `input_length=[75]`.  
- **Vocabulary**: 39 symbols → `"abcdefghijklmnopqrstuvwxyz'?!123456789 "` plus the CTC blank. 

---

## Model 

- **Input**: `(T=75, H=46, W=140, C=1)`  
- **3D Convolutions**: filters `[128, 256, 75]`, kernel `3×3×3`, pooling `(1, 2, 2)` after each block, ReLU activations  
- **Temporal**: `Bidirectional(LSTM(128)) × 2` with `Dropout(0.5)`  
- **Classifier**: `Dense(41)` + softmax  
- **CTC**: greedy decode to character sequence 

---

## Key parameters

| Name | Value |
|---|---|
| Sequence length | 75 |
| ROI crop (y, x) | 190–236, 80–220 |
| ROI size | 46 × 140 |
| Channels | 1 |
| Input shape | (75, 46, 140, 1) |
| Output classes | 41 |
| Vocabulary size | 39 (+ CTC blank) |

Numbers above are exactly what the app expects. 

---

## Citation

If you use this repository, please cite:

**Priyal Khapra, Sanskriti Agarwal, Utkarsh Sharma.** *Deep Learning for Lip Reading and Speech Recognition.* *International Journal of Science, Engineering and Technology (IJSET)*, **12**(1), 2024. [PDF](https://www.ijset.in/wp-content/uploads/IJSET_V12_issue1_575.pdf)

### BibTeX
```bibtex
@article{khapra2024deep,
  title   = {Deep Learning for Lip Reading and Speech Recognition},
  author  = {Khapra, Priyal and Agarwal, Sanskriti and Sharma, Utkarsh},
  journal = {International Journal of Science, Engineering and Technology},
  volume  = {12},
  number  = {1},
  year    = {2024},
  url     = {https://www.ijset.in/wp-content/uploads/IJSET_V12_issue1_575.pdf}
}
```