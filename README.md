# Pneumonia Detection Decision-Support Tool

An AI-powered chest X-ray analysis system built as a **Decision-Support Tool** for clinicians in areas with limited radiological expertise. The system uses a **VGG19** deep learning model trained via **Transfer Learning** on the Paul Mooney Chest X-Ray Pneumonia dataset.

> ⚠️ **Disclaimer**: This is a Decision-Support Tool. All results must be reviewed by a qualified medical professional before any clinical decision is made. This system does not replace professional radiological diagnosis.

---

## Project Structure

```
TIMO's Project/
├── training/
│   └── train_pneumonia_model.py        # GPU training script (Kaggle/Colab)
├── backend/
│   ├── app.py                          # Flask REST API
│   ├── requirements.txt                # Python dependencies
│   └── models/
│       └── pneumonia_model.h5          # Trained model weights (after training)
├── frontend/                           # React.js dashboard
│   ├── src/
│   │   ├── App.jsx                     # Main application component
│   │   └── components/
│   │       ├── Header.jsx              # Navigation header
│   │       ├── UploadSection.jsx       # X-ray upload interface
│   │       └── ResultsPanel.jsx        # Analysis results display
│   └── ...
└── README.md
```

---

## Phase 1 — Data Engineering

### Adult-Focused Dataset
- **Source**: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (Adult-Focused)
- **Classes**: NORMAL / VIRAL PNEUMONIA (binary classification)
- **Splits**: Automatic 80/20 train-validation split

### Preprocessing
- **Standardization**: All images resized to **224×224 pixels**
- **Normalization**: Pixel intensity scaled to `[0, 1]`
- **Augmentation** (training only):
  - Rotation: ±15°
  - Zoom: 15%
  - Horizontal flip
  - Width/Height shift: 10%
  - Brightness variation: 80–120%

---

## Phase 2 — Comparative Study

### Architecture: VGG19 (Transfer Learning)
The model uses VGG19 pre-trained on ImageNet as a frozen feature extractor, with a custom classification head:

```
VGG19 Base (frozen) → GlobalAveragePooling2D → BatchNorm → Dense(512) → Dropout(0.5) → BatchNorm → Dense(256) → Dropout(0.3) → Dense(1, sigmoid)
```

### Configuration 1 — Baseline
- **Loss**: Standard Binary Cross-Entropy
- **Purpose**: Establish baseline performance without addressing class imbalance

### Configuration 2 — Optimized
- **Loss**: Class-Weighted Binary Cross-Entropy
- **Class Weights**: Computed using `sklearn.utils.class_weight.compute_class_weight('balanced')`
- **Purpose**: Address the data imbalance (the "data-driven mirror") in the Kaggle dataset

### Primary Metric: **Recall**
The system is optimized for **Recall** to ensure it minimizes false negatives — a missed pneumonia case is far more dangerous than a false alarm in a clinical setting.

---

## Phase 3 — Decision-Support Deployment

### Backend (Flask API)
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Upload X-ray image for analysis |
| `GET` | `/model-info` | Model metadata and configuration |

### Frontend (React.js)
```bash
cd frontend
npm install
npm run dev
```

The frontend provides:
- Drag-and-drop X-ray image upload
- Real-time analysis with confidence scores
- Model metadata display
- Clinical disclaimer on every result

---

## Training on Kaggle

1. Create a new Kaggle Notebook
2. Add the dataset: [chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
3. Enable **GPU** accelerator (Tesla T4)
4. Copy the contents of `training/train_pneumonia_model.py` into notebook cells
5. Ensure the model file is uploaded as **pneumonia_model.zip** to the **v1.0-model** release on GitHub.
6. Download `pneumonia_model.h5` and place it in `backend/models/`

---

## Deployment & Hosting

The project is configured for automated deployment:

- **Frontend**: [Vercel](https://vercel.com)
- **Backend**: [Render](https://render.com)

### 1. Backend Setup (Render)
The backend is configured via `render.yaml`. 
1. Connect your repo to Render.
2. It will automatically detect the `backend` directory and deploy the API.
3. On first startup, it will **automatically download** the `pneumonia_model.zip` (~180MB) from the GitHub Release and extract it.

### 2. Frontend Setup (Vercel)
The frontend is configured via `frontend/vercel.json`.
1. Connect your repo to Vercel.
2. Set the **Root Directory** to `frontend`.
3. Add an **Environment Variable**: `VITE_API_URL` = Your Render API URL.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow / Keras |
| Model Architecture | VGG19 (Transfer Learning) |
| Backend API | Python Flask (Gunicorn) |
| Frontend | React.js (Vite) |
| Hosting | Vercel & Render |

---

## License

This project is for academic and research purposes.
