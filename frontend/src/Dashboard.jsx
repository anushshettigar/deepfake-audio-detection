import { useEffect, useRef, useState } from 'react'
import { useAuth } from './context/AuthContext.jsx'
import { fetchStats, fetchHistory, analyzeAudio } from './lib/api.js'
import './Dashboard.css'

function Dashboard({ onNavigate }) {
  const fileInputRef = useRef(null)
  const { isAuthenticated, loading: authLoading } = useAuth()

  const [stats, setStats] = useState({
    total_analyses: 0,
    authentic_audio: 0,
    deepfake_detected: 0,
    detection_accuracy: null
  })

  const [recentAnalyses, setRecentAnalyses] = useState([])
  const [loadingStats, setLoadingStats] = useState(true)
  const [authError, setAuthError] = useState(null)

  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const [uploadError, setUploadError] = useState('')

  useEffect(() => {
    if (!authLoading) {
      if (!isAuthenticated) {
        setAuthError('Please sign in to access the dashboard')
        setLoadingStats(false)
      } else {
        loadDashboardData()
      }
    }
  }, [isAuthenticated, authLoading])

  const loadDashboardData = async () => {
    try {
      setLoadingStats(true)
      setAuthError(null)

      const [statsData, historyData] = await Promise.all([
        fetchStats(),
        fetchHistory()
      ])

      setStats({
        total_analyses: statsData.total_analyses || 0,
        authentic_audio: statsData.authentic_audio || 0,
        deepfake_detected: statsData.deepfake_detected || 0,
        detection_accuracy: statsData.detection_accuracy
      })

      const records = Array.isArray(historyData.records) ? historyData.records : []
      setRecentAnalyses(records.slice(0, 5))

    } catch (error) {
      console.error('Dashboard loading error:', error)
      if (error.message?.includes('401') || error.message?.includes('Authentication')) {
        setAuthError('Authentication required. Please sign in again.')
      }
    } finally {
      setLoadingStats(false)
    }
  }

  const handleBrowse = () => {
    fileInputRef.current?.click()
  }

  const handleAnalyzeClick = () => {
    onNavigate?.('analyze')
  }

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0]

    if (!file) {
      return
    }

    setUploading(true)
    setUploadResult(null)
    setUploadError('')

    try {
      const data = await analyzeAudio(file)
      setUploadResult(data.analysis)
      await loadDashboardData()
    } catch (error) {
      console.error(error)
      setUploadError(error?.message || 'Unable to analyze audio')
    } finally {
      setUploading(false)
      event.target.value = ''
    }
  }

  const getLabel = (item) => {
    const label = (
      item.label ||
      item.result ||
      item.status ||
      ''
    ).toString().toUpperCase()
    if (
      label.includes('DEEPFAKE') ||
      label.includes('SPOOF')
    ) {
      return 'DEEPFAKE'
    }
    if (
      label.includes('REAL') ||
      label.includes('BONAFIDE')
    ) {
      return 'REAL'
    }
    return 'PENDING'
  }

  const formatDate = (value) => {
    if (!value) return '—'
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return value
    return date.toLocaleString()
  }

  const formatConfidence = (value) => {
    if (value === null || value === undefined) return '—'
    return `${value}%`
  }

  if (authLoading) {
    return (
      <div className="dashboard">
        <div className="loading">Loading authentication...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="dashboard">
        <div className="auth-required">
          <h2>Authentication Required</h2>
          <p>{authError || 'Please sign in to access the dashboard'}</p>
        </div>
      </div>
    )
  }

  return ( <div className="dashboard">

  <aside className="sidebar">

    <div className="brand">

      <div className="brand-mark">
        DS
      </div>

      <div>
        <strong>
          DEEPSHIELD
        </strong>

        <span>
          Audio Intelligence
        </span>
      </div>

    </div>


    <nav className="sidebar-nav">

      <button
        className="nav-item active"
        onClick={() =>
          onNavigate?.('dashboard')
        }
      >
        <span>⌂</span>
        Dashboard
      </button>


      <button
        className="nav-item"
        onClick={() =>
          onNavigate?.('analyze')
        }
      >
        <span>◉</span>
        Analyze Audio
      </button>


      <button
        className="nav-item"
        onClick={() =>
          onNavigate?.('live')
        }
      >
        <span>◌</span>
        Live Detection
      </button>


      <button
        className="nav-item"
        onClick={() =>
          onNavigate?.('history')
        }
      >
        <span>▤</span>
        Analysis History
      </button>


      <button
        className="nav-item"
        onClick={() =>
          onNavigate?.('dataset')
        }
      >
        <span>◫</span>
        Dataset
      </button>

    </nav>


    <div className="sidebar-bottom">

      <button
        className="nav-item"
        onClick={() =>
          onNavigate?.('settings')
        }
      >
        <span>⚙</span>
        Settings
      </button>

      <a
        href="/deepshield-extension.zip"
        className="nav-item download-extension"
        download="deepshield-extension.zip"
      >
        <span>⬇</span>
        Download Extension
      </a>

      <div className="system-status">

        <span className="status-dot" />

        <div>
          <strong>
            Detection Engine
          </strong>

          <small>
            System ready
          </small>
        </div>

      </div>

    </div>

  </aside>


  <section className="main-content">

    <header className="topbar">

      <div>

        <p className="eyebrow">
          SECURITY ANALYTICS
        </p>

        <h1>
          Audio Detection Dashboard
        </h1>

      </div>


      <div className="topbar-actions">

        <button
          className="icon-button"
          aria-label="Search"
        >
          ⌕
        </button>


        <button
          className="icon-button"
          aria-label="Appearance"
        >
          ◔
        </button>


        <div className="profile">

          <div className="profile-avatar">
            A
          </div>

          <div>

            <strong>
              Administrator
            </strong>

            <span>
              Secure session
            </span>

          </div>

        </div>

      </div>

    </header>


    <div className="dashboard-content">

      <section className="hero-card">

        <div className="hero-copy">

          <span className="live-badge">
            <span />
            ENGINE ONLINE
          </span>


          <h2>
            Detect synthetic
            <br />
            <span>
              voices with confidence.
            </span>
          </h2>


          <p>
            Upload an audio recording and
            analyze it using the DEEPSHIELD
            detection engine.
          </p>


          <button
            className="primary-button"
            onClick={handleAnalyzeClick}
          >
            <span>+</span>
            Analyze Audio
          </button>

        </div>


        <div className="hero-visual">

          <div className="visual-ring ring-one" />
          <div className="visual-ring ring-two" />
          <div className="visual-ring ring-three" />

          <div className="visual-core">
            <div className="core-line" />
            <div className="core-line" />
            <div className="core-line" />
          </div>

          <div className="signal signal-one" />
          <div className="signal signal-two" />
          <div className="signal signal-three" />

        </div>

      </section>


      <section className="stats-grid">

        <article className="stat-card">

          <div className="stat-icon">
            ◉
          </div>

          <div>

            <span>
              Total Analyses
            </span>

            <strong>
              {loadingStats
                ? '—'
                : stats.total_analyses}
            </strong>

          </div>

          <small>
            All analyzed recordings
          </small>

        </article>


        <article className="stat-card">

          <div className="stat-icon">
            ✓
          </div>

          <div>

            <span>
              Authentic Audio
            </span>

            <strong>
              {loadingStats
                ? '—'
                : stats.authentic_audio}
            </strong>

          </div>

          <small>
            Verified recordings
          </small>

        </article>


        <article className="stat-card">

          <div className="stat-icon">
            !
          </div>

          <div>

            <span>
              Deepfake Detected
            </span>

            <strong>
              {loadingStats
                ? '—'
                : stats.deepfake_detected}
            </strong>

          </div>

          <small>
            Suspicious recordings
          </small>

        </article>


        <article className="stat-card">

          <div className="stat-icon">
            ◌
          </div>

          <div>

            <span>
              Detection Accuracy
            </span>

            <strong>
              {stats.detection_accuracy !==
                null &&
              stats.detection_accuracy !==
                undefined
                ? `${Number(
                    stats.detection_accuracy
                  ).toFixed(2)}%`
                : '—'}
            </strong>

          </div>

          <small>
            Average model confidence
          </small>

        </article>

      </section>


      <section className="workspace-grid">

        <article className="workspace-card">

          <div className="card-heading">

            <div>

              <span className="card-label">
                QUICK ANALYSIS
              </span>

              <h3>
                Upload an audio file
              </h3>

            </div>

            <span className="secure-label">
              SECURE
            </span>

          </div>


          <div className="upload-zone">

            <div className="upload-icon">
              ↑
            </div>


            <h4>
              Drop your audio here
            </h4>


            <p>
              or choose a file from your device
            </p>


            <input
              ref={fileInputRef}
              type="file"
              accept=".wav,.mp3,.flac,.m4a,.ogg,audio/*"
              onChange={handleFileChange}
              hidden
            />


            <button
              className="secondary-button"
              onClick={handleBrowse}
              disabled={uploading}
            >
              {uploading
                ? 'Analyzing...'
                : 'Browse Files'}
            </button>


            {uploadResult && (

              <span className="upload-success">

                {getLabel(
                  uploadResult
                )}

                {' · '}

                {formatConfidence(
                  uploadResult.confidence
                )}

              </span>

            )}


            {uploadError && (

              <span className="upload-error">
                {uploadError}
              </span>

            )}


            <span className="file-hint">
              WAV · MP3 · FLAC · M4A · OGG
            </span>

          </div>

        </article>


        <article className="workspace-card activity-card">

          <div className="card-heading">

            <div>

              <span className="card-label">
                SYSTEM ACTIVITY
              </span>

              <h3>
                Recent analyses
              </h3>

            </div>


            <button
              className="text-button"
              onClick={() =>
                onNavigate?.('history')
              }
            >
              View all
            </button>

          </div>


          {recentAnalyses.length === 0 ? (

            <div className="empty-state">

              <div className="empty-icon">
                ◌
              </div>

              <strong>
                No analyses yet
              </strong>

              <span>
                Your detection activity
                will appear here.
              </span>

            </div>

          ) : (

            <div className="recent-analysis-list">

              {recentAnalyses.map(
                (item, index) => {

                  const label =
                    getLabel(item)

                  return (

                    <div
                      className="recent-analysis-item"
                      key={
                        item.id ||
                        index
                      }
                    >

                      <div className="recent-audio-icon">
                        ♪
                      </div>


                      <div className="recent-analysis-info">

                        <strong>
                          {item.filename ||
                            'Audio recording'}
                        </strong>

                        <span>
                          {formatDate(
                            item.created_at
                          )}
                        </span>

                      </div>


                      <div className="recent-analysis-result">

                        <strong
                          className={
                            label === 'DEEPFAKE'
                              ? 'deepfake'
                              : label === 'REAL'
                                ? 'real'
                                : 'pending'
                          }
                        >
                          {label}
                        </strong>

                        <span>
                          {formatConfidence(
                            item.confidence
                          )}
                        </span>

                      </div>

                    </div>

                  )
                }
              )}

            </div>

          )}

        </article>

      </section>

    </div>

  </section>

</div>
)
}

export default Dashboard