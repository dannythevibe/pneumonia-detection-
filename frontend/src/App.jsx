import { useState } from 'react'
import Header from './components/Header'
import UploadSection from './components/UploadSection'
import ResultsPanel from './components/ResultsPanel'
import LandingPage from './components/LandingPage'
import './index.css'

function App() {
  const [showApp, setShowApp] = useState(false)
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [result, setResult] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState(null)

  const handleImageSelect = (file) => {
    setSelectedImage(file)
    setResult(null)
    setError(null)

    const reader = new FileReader()
    reader.onload = (e) => setImagePreview(e.target.result)
    reader.readAsDataURL(file)
  }

  const handleAnalyze = async () => {
    if (!selectedImage) return

    setIsAnalyzing(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('image', selectedImage)

    try {
      const API = import.meta.env.VITE_API_URL || 'http://localhost:5001'
      const response = await fetch(`${API}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || 'Prediction failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to connect to the server. Ensure the Flask backend is running.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleReset = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setResult(null)
    setError(null)
  }

  if (!showApp) {
    return <LandingPage onGetStarted={() => setShowApp(true)} />
  }

  return (
    <>
      <Header onHome={() => setShowApp(false)} />
      <main className="main-content">
        <div className="app-container">
          <div className="page-hero">
            <h1>Chest X-Ray <span>Analysis</span></h1>
            <p>
              Upload a chest X-ray for automated pneumonia screening.
              AI-powered preliminary analysis trained on both pediatric and adult data.
            </p>
          </div>

          <div className="dashboard-grid">
            <UploadSection
              imagePreview={imagePreview}
              selectedImage={selectedImage}
              isAnalyzing={isAnalyzing}
              onImageSelect={handleImageSelect}
              onAnalyze={handleAnalyze}
              onReset={handleReset}
            />

            <ResultsPanel
              result={result}
              isAnalyzing={isAnalyzing}
              error={error}
            />
          </div>
        </div>
      </main>

      <footer className="footer">
        <div className="app-container">
          <p>
            Pneumonia Detection Decision-Support Tool &nbsp;|&nbsp;
            Neural Analysis System V1.0
          </p>
        </div>
      </footer>
    </>
  )
}

export default App
