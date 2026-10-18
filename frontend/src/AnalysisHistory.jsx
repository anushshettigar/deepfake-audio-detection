import { useEffect, useState } from 'react'
import { useAuth } from './context/AuthContext.jsx'
import { fetchHistory } from './lib/api.js'
import './AnalysisHistory.css'

function AnalysisHistory({ onNavigate }) {
  const { isAuthenticated, loading: authLoading } = useAuth()
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [authError, setAuthError] = useState(null)

  const loadHistory = async () => {
    try {
      setLoading(true)
      setError('')
      setAuthError(null)

      const data = await fetchHistory()

      setHistory(
        Array.isArray(data.records)
          ? data.records
          : []
      )

    } catch (err) {
      console.error(err)
      setError(err.message || 'Unable to load history.')
      if (err.message?.includes('401') || err.message?.includes('Authentication')) {
        setAuthError('Authentication required. Please sign in again.')
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      loadHistory()
    }
  }, [isAuthenticated, authLoading])

  if (authLoading) {
    return (
      <div className="history-page">
        <div className="loading">Loading authentication...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="history-page">
        <div className="auth-required">
          <h2>Authentication Required</h2>
          <p>{authError || 'Please sign in to view history'}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="history-page">

      <header className="history-header">

        <div>

          <button
            className="back-button"
            onClick={() => onNavigate?.('dashboard')}
          >
            ← Back
          </button>

          <span className="history-eyebrow">
            SECURITY ANALYTICS
          </span>

          <h1>Analysis History</h1>

          <p>
            Review previous audio detection results.
          </p>

        </div>

        <button
          className="refresh-button"
          onClick={loadHistory}
          disabled={loading}
        >
          {loading ? 'Loading...' : '↻ Refresh'}
        </button>

      </header>

      <main className="history-content">

        {error && (
          <div className="history-error">
            <strong>
              Unable to load history
            </strong>

            <span>
              {error}
            </span>
          </div>
        )}

        {loading && !error && (
          <div className="history-empty">

            <div className="history-empty-icon">
              ◌
            </div>

            <h2>
              Loading history...
            </h2>

            <p>
              Checking your previous analyses.
            </p>

          </div>
        )}

        {!loading &&
          !error &&
          history.length === 0 && (

          <div className="history-empty">

            <div className="history-empty-icon">
              ◌
            </div>

            <h2>
              No analyses yet
            </h2>

            <p>
              Upload an audio file from
              Analyze Audio and your detection
              results will appear here.
            </p>

            <button
              className="history-primary-button"
              onClick={() =>
                onNavigate?.('analyze')
              }
            >
              Analyze Audio
            </button>

          </div>
        )}

        {!loading &&
          !error &&
          history.length > 0 && (

          <div className="history-list">

            {history.map((item, index) => {

              const label =
                item.label || 'PENDING'

              const confidence =
                item.confidence !== null &&
                item.confidence !== undefined
                  ? `${Number(item.confidence).toFixed(1)}%`
                  : '—'

              const isDeepfake =
                String(label)
                  .toUpperCase()
                  .includes('DEEPFAKE')

              return (
                <article
                  className="history-card"
                  key={
                    item.id ||
                    `${index}-${item.filename}`
                  }
                >

                  <div className="history-card-icon">
                    {isDeepfake ? '!' : '✓'}
                  </div>

                  <div className="history-card-main">

                    <div className="history-card-top">

                      <div>

                        <h3>
                          {item.filename ||
                            `Audio Analysis #${index + 1}`}
                        </h3>

                        <span>
                          {item.created_at
                            ? new Date(
                                item.created_at
                              ).toLocaleString()
                            : 'Analysis recorded'}
                        </span>

                      </div>

                      <div className="history-result">
                        {label}
                      </div>

                    </div>

                    <div className="history-card-details">

                      <div>
                        <span>
                          Confidence
                        </span>

                        <strong>
                          {confidence}
                        </strong>
                      </div>

                      <div>
                        <span>
                          Status
                        </span>

                        <strong>
                          {item.status || 'PENDING'}
                        </strong>
                      </div>

                      <div>
                        <span>
                          Format
                        </span>

                        <strong>
                          {item.format
                            ? item.format.toUpperCase()
                            : 'Audio'}
                        </strong>
                      </div>

                      <div>
                        <span>
                          Size
                        </span>

                        <strong>
                          {item.size_bytes
                            ? `${(
                                item.size_bytes /
                                1024 /
                                1024
                              ).toFixed(2)} MB`
                            : '—'}
                        </strong>
                      </div>

                    </div>

                  </div>

                </article>
              )
            })}

          </div>
        )}

      </main>

    </div>
  )
}

export default AnalysisHistory