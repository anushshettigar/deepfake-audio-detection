import { useRef, useState } from 'react'
import { useAuth } from './context/AuthContext.jsx'
import { analyzeAudio } from './lib/api.js'
import './AnalyzeAudio.css'

function AnalyzeAudio({ onNavigate }) {
  const { isAuthenticated, loading: authLoading } = useAuth()
  const fileInputRef = useRef(null)

  const [file, setFile] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [authError, setAuthError] = useState(null)

  const handleBrowse = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0]

    if (!selectedFile) {
      return
    }

    setFile(selectedFile)
    setResult(null)
    setError('')

    event.target.value = ''
  }

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select an audio file first.')
      return
    }

    setIsAnalyzing(true)
    setResult(null)
    setError('')

    try {
      const data = await analyzeAudio(file)
      setResult(data.analysis)
    } catch (err) {
      console.error(err)
      setError(err.message || 'Unable to analyze audio.')
      if (err.message?.includes('401') || err.message?.includes('Authentication')) {
        setAuthError('Authentication required. Please sign in again.')
      }
    } finally {
      setIsAnalyzing(false)
    }
  }

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B'
    const mb = bytes / (1024 * 1024)
    if (mb >= 1) return `${mb.toFixed(2)} MB`
    const kb = bytes / 1024
    return `${kb.toFixed(1)} KB`
  }

  const isDeepfake = result?.label === 'DEEPFAKE'
  const isReal = result?.label === 'REAL'

  if (authLoading) {
    return (
      <div className="analyze-page">
        <div className="loading">Loading authentication...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="analyze-page">
        <div className="auth-required">
          <h2>Authentication Required</h2>
          <p>{authError || 'Please sign in to analyze audio'}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="analyze-page">

      <header className="analyze-header">

        <button
          className="back-button"
          onClick={() =>
            onNavigate?.('dashboard')
          }
        >
          ← Back to Dashboard
        </button>

        <div className="analyze-title">

          <span>
            AUDIO FORENSICS
          </span>

          <h1>
            Analyze Audio
          </h1>

          <p>
            Upload an audio recording for
            DEEPSHIELD analysis.
          </p>

        </div>

      </header>

      <main className="analyze-content">

        <section className="analyze-upload-card">

          <div className="section-label">
            AUDIO INPUT
          </div>

          <h2>
            Upload an audio recording
          </h2>

          <p className="section-description">
            Select an audio file from your
            device to begin analysis.
          </p>

          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.flac,.m4a,.ogg,.webm,audio/*"
            onChange={handleFileChange}
            hidden
          />

          <div
            className={
              `large-upload-zone ${
                file
                  ? 'has-file'
                  : ''
              }`
            }
            onClick={handleBrowse}
          >

            <div className="upload-symbol">
              {file ? '✓' : '↑'}
            </div>

            {file ? (
              <>
                <strong>
                  {file.name}
                </strong>

                <span>
                  {formatFileSize(
                    file.size
                  )}
                </span>
              </>
            ) : (
              <>
                <strong>
                  Drop your audio here
                </strong>

                <span>
                  or click to browse
                </span>
              </>
            )}

          </div>

          <div className="format-row">

            <span>
              WAV
            </span>

            <span>
              MP3
            </span>

            <span>
              FLAC
            </span>

            <span>
              M4A
            </span>

            <span>
              OGG
            </span>

            <span>
              MAX 50 MB
            </span>

          </div>

          {file && (
            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={isAnalyzing}
            >
              {isAnalyzing
                ? 'Analyzing...'
                : 'Analyze Recording'}
            </button>
          )}

          {error && (
            <div className="analysis-error">
              {error}
            </div>
          )}

        </section>

        {isAnalyzing && (
          <section className="processing-card">

            <div className="processing-spinner" />

            <div>
              <strong>
                Processing audio
              </strong>

              <span>
                Analyzing recording with the
                DEEPSHIELD detection model...
              </span>
            </div>

          </section>
        )}

        {result && (
          <section className="result-card">

            <div className="section-label">
              ANALYSIS RESULT
            </div>

            <div className="result-main">

              <div
                className={
                  `result-icon ${
                    isDeepfake
                      ? 'pending'
                      : ''
                  }`
                }
              >
                {isDeepfake
                  ? '!'
                  : '✓'}
              </div>

              <div className="result-copy">

                <span>
                  DETECTION STATUS
                </span>

                <h2>
                  COMPLETED
                </h2>

                <p>
                  {isDeepfake
                    ? 'The detection model identified this recording as synthetic audio.'
                    : 'The detection model identified this recording as authentic audio.'}
                </p>

              </div>

            </div>

            <div className="result-details">

              <div>
                <span>
                  File
                </span>

                <strong>
                  {result.filename}
                </strong>
              </div>

              <div>
                <span>
                  Format
                </span>

                <strong>
                  {result.format?.toUpperCase()}
                </strong>
              </div>

              <div>
                <span>
                  Size
                </span>

                <strong>
                  {formatFileSize(
                    result.size_bytes
                  )}
                </strong>
              </div>

              <div>
                <span>
                  Confidence
                </span>

                <strong>
                  {result.confidence !== undefined
                    ? `${result.confidence}%`
                    : '—'}
                </strong>
              </div>

            </div>

            <div className="pending-notice">

              <strong>
                {isReal
                  ? 'REAL AUDIO'
                  : isDeepfake
                    ? 'DEEPFAKE DETECTED'
                    : 'UNKNOWN RESULT'}
              </strong>

              <span>
                {isReal
                  ? 'The trained ASVspoof model classified this recording as REAL.'
                  : isDeepfake
                    ? 'The trained ASVspoof model classified this recording as DEEPFAKE.'
                    : 'The detection model returned an unexpected classification.'}
              </span>

            </div>

            <div className="result-actions">

              <button
                onClick={() =>
                  onNavigate?.('history')
                }
              >
                View Analysis History
              </button>

              <button
                onClick={() => {
                  setFile(null)
                  setResult(null)
                  setError('')
                }}
              >
                Analyze Another File
              </button>

            </div>

          </section>
        )}

      </main>

    </div>
  )
}

export default AnalyzeAudio