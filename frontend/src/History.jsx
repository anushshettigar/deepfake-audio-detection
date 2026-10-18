import { useEffect, useState } from 'react'
import './History.css'

function History({ onNavigate }) {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    loadHistory()
  }, [])

  const loadHistory = async () => {
    try {
      setLoading(true)
      setError('')

      const response = await fetch('/api/history')

      if (!response.ok) {
        throw new Error('Unable to load analysis history')
      }

      const data = await response.json()

      /*
       * Backend returns:
       * {
       *   status: "success",
       *   count: number,
       *   records: [...]
       * }
       */
      if (Array.isArray(data.records)) {
        setHistory(data.records)
      } else if (Array.isArray(data.history)) {
        setHistory(data.history)
      } else if (Array.isArray(data)) {
        setHistory(data)
      } else {
        setHistory([])
      }

    } catch (error) {
      console.error('History loading error:', error)

      setError(
        error?.message ||
        'Unable to load analysis history.'
      )

      setHistory([])

    } finally {
      setLoading(false)
    }
  }

  const clearHistory = async () => {
    const confirmed = window.confirm(
      'Are you sure you want to clear all analysis history?'
    )

    if (!confirmed) {
      return
    }

    try {
      const response = await fetch(
        '/api/history',
        {
          method: 'DELETE'
        }
      )

      if (!response.ok) {
        throw new Error(
          'Unable to clear analysis history'
        )
      }

      setHistory([])
      setError('')

    } catch (error) {
      console.error(error)

      setError(
        error?.message ||
        'Unable to clear analysis history.'
      )
    }
  }

  const getLabel = (item) => {
    const label = (
      item.label ||
      item.result ||
      item.status ||
      ''
    )
      .toString()
      .toLowerCase()

    if (
      label.includes('fake') ||
      label.includes('spoof') ||
      label.includes('deepfake')
    ) {
      return 'DEEPFAKE'
    }

    if (
      label.includes('real') ||
      label.includes('bonafide') ||
      label.includes('authentic')
    ) {
      return 'REAL'
    }

    return 'PENDING'
  }

  const getConfidence = (item) => {
    if (
      item.confidence === undefined ||
      item.confidence === null
    ) {
      return '—'
    }

    return `${item.confidence}%`
  }

  const getFilename = (item) => {
    return (
      item.filename ||
      item.file_name ||
      item.name ||
      'Audio recording'
    )
  }

  const getSource = (item) => {
    if (item.source === 'extension_live') {
      return 'LIVE EXTENSION'
    }

    if (item.source === 'upload') {
      return 'AUDIO UPLOAD'
    }

    return item.source || 'Audio Upload'
  }

  const formatDate = (item) => {
    const value =
      item.created_at ||
      item.timestamp ||
      item.date

    if (!value) {
      return '—'
    }

    const date = new Date(value)

    if (Number.isNaN(date.getTime())) {
      return value
    }

    return date.toLocaleString()
  }

  return (
    <div className="history-page">

      <header className="history-header">

        <div className="history-header-left">

          <button
            className="back-button"
            onClick={() =>
              onNavigate?.('dashboard')
            }
          >
            ←
            <span>Back</span>
          </button>

          <div>

            <span className="history-eyebrow">
              SECURITY ANALYTICS
            </span>

            <h1>
              Analysis History
            </h1>

            <p>
              Review previous audio detection results.
            </p>

          </div>

        </div>

        <button
          className="clear-history-button"
          onClick={clearHistory}
          disabled={history.length === 0}
        >
          Clear History
        </button>

      </header>

      <main className="history-content">

        <section className="history-summary">

          <div className="history-summary-card">
            <span>Total Analyses</span>
            <strong>
              {history.length}
            </strong>
          </div>

          <div className="history-summary-card">
            <span>Authentic</span>

            <strong>
              {
                history.filter(
                  item =>
                    getLabel(item) === 'REAL'
                ).length
              }
            </strong>
          </div>

          <div className="history-summary-card">
            <span>Deepfakes</span>

            <strong>
              {
                history.filter(
                  item =>
                    getLabel(item) === 'DEEPFAKE'
                ).length
              }
            </strong>
          </div>

          <div className="history-summary-card">
            <span>Pending</span>

            <strong>
              {
                history.filter(
                  item =>
                    getLabel(item) === 'PENDING'
                ).length
              }
            </strong>
          </div>

        </section>

        <section className="history-panel">

          <div className="history-panel-heading">

            <div>

              <span className="history-label">
                DETECTION RECORDS
              </span>

              <h2>
                Previous analyses
              </h2>

            </div>

            <button
              className="new-analysis-button"
              onClick={() =>
                onNavigate?.('analyze')
              }
            >
              + New Analysis
            </button>

          </div>

          {error && (
            <div className="history-empty">
              <div className="history-empty-icon">
                !
              </div>

              <strong>
                Unable to load history
              </strong>

              <span>
                {error}
              </span>

              <button
                className="empty-action-button"
                onClick={loadHistory}
              >
                Try Again
              </button>
            </div>
          )}

          {!error && loading ? (
            <div className="history-empty">

              <div className="history-empty-icon">
                ◌
              </div>

              <strong>
                Loading history...
              </strong>

              <span>
                Retrieving your previous detection records.
              </span>

            </div>

          ) : !error && !loading && history.length === 0 ? (

            <div className="history-empty">

              <div className="history-empty-icon">
                ◌
              </div>

              <strong>
                No analyses yet
              </strong>

              <span>
                Your audio detection results will appear
                here after you analyze a recording.
              </span>

              <button
                className="empty-action-button"
                onClick={() =>
                  onNavigate?.('analyze')
                }
              >
                Analyze Audio
              </button>

            </div>

          ) : !error && !loading ? (

            <div className="history-table-wrapper">

              <table className="history-table">

                <thead>

                  <tr>
                    <th>Audio</th>
                    <th>Source</th>
                    <th>Result</th>
                    <th>Confidence</th>
                    <th>Date</th>
                  </tr>

                </thead>

                <tbody>

                  {history.map(
                    (item, index) => {

                      const label =
                        getLabel(item)

                      return (

                        <tr
                          key={
                            item.id ||
                            index
                          }
                        >

                          <td>

                            <div className="audio-name">

                              <div className="audio-icon">
                                ♪
                              </div>

                              <div>

                                <strong>
                                  {getFilename(item)}
                                </strong>

                                <span>
                                  {item.source ===
                                  'extension_live'
                                    ? `Session ${
                                        item.session_id
                                          ? item.session_id.slice(
                                              0,
                                              8
                                            )
                                          : '—'
                                      }`
                                    : 'Audio recording'}
                                </span>

                              </div>

                            </div>

                          </td>

                          <td>

                            <span className="source-badge">
                              {getSource(item)}
                            </span>

                          </td>

                          <td>

                            <span
                              className={`result-badge ${label.toLowerCase()}`}
                            >
                              <span />
                              {label}
                            </span>

                          </td>

                          <td>

                            <strong className="confidence">
                              {getConfidence(item)}
                            </strong>

                          </td>

                          <td>

                            <span className="history-date">
                              {formatDate(item)}
                            </span>

                          </td>

                        </tr>

                      )
                    }
                  )}

                </tbody>

              </table>

            </div>

          ) : null}

        </section>

      </main>

    </div>
  )
}

export default History