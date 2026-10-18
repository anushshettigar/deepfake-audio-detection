import { useState, useEffect } from 'react'
import { useAuth } from './context/AuthContext.jsx'
import { fetchTrainingDataset } from './lib/api.js'
import './Dataset.css'

function Dataset({ onNavigate }) {
  const { isAuthenticated, loading: authLoading } = useAuth()
  const [datasetStatus, setDatasetStatus] = useState('Ready')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [authError, setAuthError] = useState(null)

  const loadDataset = async () => {
    try {
      setLoading(true)
      setError('')
      setAuthError(null)

      const result = await fetchTrainingDataset()
      setData(result)
      setDatasetStatus('Loaded')
    } catch (err) {
      console.error(err)
      setError(err.message || 'Unable to load dataset.')
      if (err.message?.includes('401') || err.message?.includes('Authentication')) {
        setAuthError('Authentication required. Please sign in again.')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleBack = () => {
    onNavigate?.('dashboard')
  }

  if (authLoading) {
    return (
      <div className="dataset-page">
        <div className="loading">Loading authentication...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="dataset-page">
        <div className="auth-required">
          <h2>Authentication Required</h2>
          <p>{authError || 'Please sign in to view dataset'}</p>
        </div>
      </div>
    )
  }

  const summary = data?.summary || {}
  const datasets = data?.datasets || {}

  return (
    <div className="dataset-page">

      <header className="dataset-header">

        <div>
          <button
            className="dataset-back"
            onClick={handleBack}
          >
            ← Back
          </button>

          <span className="dataset-eyebrow">
            MODEL DATA
          </span>

          <h1>Training Datasets</h1>

          <p>
            {summary.datasets_count || 0} datasets · {summary.total_samples?.toLocaleString() || '—'} total samples ·
            {summary.total_real?.toLocaleString() || '—'} REAL · {summary.total_fake?.toLocaleString() || '—'} FAKE
          </p>
        </div>

        <div className="dataset-status">
          <span />
          {datasetStatus}
        </div>

      </header>

      <main className="dataset-content">

        <section className="dataset-overview">

          <div className="dataset-icon">
            ◫
          </div>

          <div>
            <span className="dataset-label">
              TRAINING DATA OVERVIEW
            </span>

            <h2>All Datasets Combined</h2>

            <p>
              {summary.total_samples?.toLocaleString() || '—'} total samples ·
              {summary.total_real?.toLocaleString() || '—'} REAL (bonafide) ·
              {summary.total_fake?.toLocaleString() || '—'} FAKE (spoof)
            </p>
          </div>

        </section>

        <section className="dataset-grid">

          <article className="dataset-card">
            <span>TOTAL SAMPLES</span>
            <strong>{summary.total_samples?.toLocaleString() ?? '—'}</strong>
            <small>All datasets combined</small>
          </article>

          <article className="dataset-card">
            <span>REAL AUDIO</span>
            <strong>{summary.total_real?.toLocaleString() ?? '—'}</strong>
            <small>Bonafide (human) speech</small>
          </article>

          <article className="dataset-card">
            <span>SPOOF AUDIO</span>
            <strong>{summary.total_fake?.toLocaleString() ?? '—'}</strong>
            <small>Deepfake (TTS/VC) speech</small>
          </article>

          <article className="dataset-card">
            <span>DATASETS</span>
            <strong>{summary.datasets_count ?? 0}</strong>
            <small>Available datasets</small>
          </article>

        </section>

        <section className="dataset-info">

          <h3>Individual Datasets</h3>

          <div className="datasets-container">
            {Object.entries(datasets).map(([dsName, dsData]) => (
              <div key={dsName} className="dataset-section">
                <div className="dataset-section-header">
                  <h4>{dsName}</h4>
                  <p className="split-desc">{dsData.description}</p>
                </div>
                <div className="splits-container">
                  {Object.entries(dsData.splits).map(([splitType, splitStats]) => (
                    <div key={splitType} className="split-section">
                      <h5>{splitType.replace('_', ' ')}</h5>
                      <div className="split-grid">
                        {Object.entries(splitStats).map(([split, stats]) => (
                          <div key={split} className="split-card">
                            <span className="split-name">{split.charAt(0).toUpperCase() + split.slice(1)}</span>
                            <div className="split-stats">
                              <span>REAL: {stats.real?.toLocaleString() ?? 0}</span>
                              <span>FAKE: {stats.fake?.toLocaleString() ?? 0}</span>
                              <span>Total: {stats.total?.toLocaleString() ?? 0}</span>
                              {stats.speakers !== undefined && <span>Speakers: {stats.speakers ?? 0}</span>}
                              {stats.generators && stats.generators.length > 0 && (
                                <span className="generators">Generators: {stats.generators.join(', ')}</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

        </section>

        <section className="dataset-notice">

          <div>
            <strong>Dataset Pipeline</strong>

            <p>
              Raw Audio → Preprocessing (16kHz, 4s) → RawNet2 Features → Training
            </p>
          </div>

          <button
            onClick={loadDataset}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Refresh Dataset'}
          </button>

          {error && (
            <div className="dataset-error">
              {error}
            </div>
          )}

        </section>

      </main>

    </div>
  )
}

export default Dataset