# Modern Hosting Guide (Vercel + Render)

The project is now optimized for automated hosting. The backend handles the large model file by downloading it on-demand from GitHub Releases.

## 1. Backend (Flask API) → **Render**

The backend is pre-configured with `render.yaml`.

### Steps:
1.  Log in to [Render.com](https://render.com/).
2.  Click **"New"** → **"Blueprint"** (or connect your repo).
3.  Render will read `render.yaml` and create the `pneumonia-detection-api` service.
4.  **Auto-Download**: The server will automatically fetch the `pneumonia_model.zip` (~180MB) from GitHub on its first run and extract it.

---

## 2. Frontend (React/Vite) → **Vercel**

### Steps:
1.  Go to [Vercel](https://vercel.com/) and import your repository.
2.  Set the **Root Directory** to `frontend`.
3.  **Environment Variables**:
    *   Add `VITE_API_URL` and set it to your Render service URL (e.g., `https://pneumonia-api.onrender.com`).
4.  Click **Deploy**.

---

## 3. GitHub Release (Important)

The backend expects the model to be available at:
`https://github.com/dannythevibe/pneumonia-detection-/releases/download/v1.0-model/pneumonia_model.zip`

1. Ensure the model file is uploaded as **pneumonia_model.zip** to the **v1.0-model** release on GitHub.
