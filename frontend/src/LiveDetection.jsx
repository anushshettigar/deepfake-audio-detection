import { useEffect, useRef, useState } from 'react'
import { useAuth } from './context/AuthContext.jsx'
import { startLiveSession, analyzeLiveChunk, stopLiveSession } from './lib/api.js'
import './LiveDetection.css'

const CHUNK_DURATION = 3000

function LiveDetection({ onNavigate }) {
  const { isAuthenticated, loading: authLoading } = useAuth()
  const [isDetecting, setIsDetecting] = useState(false)
  const [status, setStatus] = useState('Ready')
  const [error, setError] = useState('')
  const [chunksSent, setChunksSent] = useState(0)
  const [result, setResult] = useState(null)
  const [authError, setAuthError] = useState(null)

  const streamRef = useRef(null)
  const recorderRef = useRef(null)
  const sessionIdRef = useRef(null)
  const activeRef = useRef(false)
  const stoppingRef = useRef(false)
  const chunkRef = useRef(0)
  const timerRef = useRef(null)

  useEffect(() => {
    return () => {
      stoppingRef.current = true
      activeRef.current = false

      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }

      const recorder = recorderRef.current

      if (recorder && recorder.state !== 'inactive') {
        try {
          recorder.stop()
        } catch {}
      }

      const stream = streamRef.current

      if (stream) {
        stream.getTracks().forEach((track) => {
          try {
            track.stop()
          } catch {}
        })
      }
    }
  }, [])

  const getMimeType = () => {
    const types = [
      'audio/webm;codecs=opus',
      'audio/webm'
    ]

    return types.find((type) =>
      MediaRecorder.isTypeSupported(type)
    )
  }

  const startDetection = async () => {
    if (activeRef.current) {
      return
    }

    setError('')
    setAuthError(null)
    setResult(null)
    setChunksSent(0)
    setStatus('Requesting microphone access...')

    activeRef.current = true
    stoppingRef.current = false
    chunkRef.current = 0

    try {
      if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
      ) {
        throw new Error(
          'Audio capture is not supported by this browser.'
        )
      }

      if (
        typeof MediaRecorder === 'undefined'
      ) {
        throw new Error(
          'MediaRecorder is not supported by this browser.'
        )
      }

      const mimeType = getMimeType()

      if (!mimeType) {
        throw new Error(
          'No compatible WebM audio recording format is supported.'
        )
      }

      const stream =
        await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false
          }
        })

      if (!activeRef.current) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }

      streamRef.current = stream

      const sessionData = await startLiveSession()

      if (!sessionData.session_id) {
        throw new Error(
          'Backend did not return a session ID.'
        )
      }

      sessionIdRef.current = sessionData.session_id

      setIsDetecting(true)
      setStatus('Listening...')

      console.log(
        'DEEPSHIELD WEB LIVE SESSION:',
        sessionData.session_id
      )

      recordChunk(mimeType)

    } catch (err) {
      console.error(
        'LIVE DETECTION START ERROR:',
        err
      )

      activeRef.current = false
      stoppingRef.current = true

      setIsDetecting(false)
      setStatus('Ready')

      setError(
        err?.message ||
        'Unable to start live detection.'
      )

      if (err.message?.includes('401') || err.message?.includes('Authentication')) {
        setAuthError('Authentication required. Please sign in again.')
      }

      await releaseResources()
    }
  }

  const recordChunk = (mimeType) => {
    if (!activeRef.current) {
      return
    }

    if (stoppingRef.current) {
      return
    }

    const stream = streamRef.current

    if (!stream) {
      return
    }

    chunkRef.current += 1

    const chunkNumber = chunkRef.current

    console.log(
      `DEEPSHIELD: STARTING CHUNK ${chunkNumber}`
    )

    const recorder = new MediaRecorder(stream, { mimeType })

    recorderRef.current = recorder

    const parts = []

    let stoppedNormally = false

    recorder.onstart = () => {
      if (!activeRef.current) {
        return
      }

      setIsDetecting(true)

      setStatus(`Listening... Chunk ${chunkNumber}`)

      console.log(
        `DEEPSHIELD: RECORDER STARTED CHUNK ${chunkNumber}`
      )
    }

    recorder.ondataavailable = (event) => {
      if (
        event.data &&
        event.data.size > 0
      ) {
        parts.push(event.data)

        console.log(
          `DEEPSHIELD: CHUNK ${chunkNumber} DATA ${event.data.size} BYTES`
        )
      }
    }

    recorder.onerror = (event) => {
      console.error(
        `DEEPSHIELD: RECORDER ERROR CHUNK ${chunkNumber}`,
        event
      )

      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }

      if (activeRef.current) {
        setError('Audio recorder encountered an error.')
        setStatus('Error')
      }

      activeRef.current = false
      stoppingRef.current = true
    }

    recorder.onstop = async () => {
      console.log(
        `DEEPSHIELD: RECORDER STOPPED CHUNK ${chunkNumber}`
      )

      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }

      if (recorderRef.current === recorder) {
        recorderRef.current = null
      }

      if (!stoppedNormally) {
        return
      }

      const blob = new Blob(parts, { type: mimeType })

      console.log(
        `DEEPSHIELD: COMPLETE CHUNK ${chunkNumber}: ${blob.size} BYTES`
      )

      if (blob.size < 1000) {
        console.warn(`DEEPSHIELD: CHUNK ${chunkNumber} TOO SMALL`)

        if (activeRef.current && !stoppingRef.current) {
          setTimeout(() => {
            recordChunk(mimeType)
          }, 200)
        }

        return
      }

      if (!activeRef.current || stoppingRef.current) {
        return
      }

      await sendChunk(blob, chunkNumber)

      if (activeRef.current && !stoppingRef.current) {
        setTimeout(() => {
          recordChunk(mimeType)
        }, 150)
      }
    }

    try {
      recorder.start()

      timerRef.current = setTimeout(() => {
        if (
          recorder.state !== 'inactive' &&
          activeRef.current &&
          !stoppingRef.current
        ) {
          stoppedNormally = true

          console.log(`DEEPSHIELD: STOPPING CHUNK ${chunkNumber}`)

          try {
            recorder.stop()
          } catch (err) {
            console.error('RECORDER STOP ERROR:', err)
          }
        }
      }, CHUNK_DURATION)

    } catch (err) {
      console.error('RECORDER START ERROR:', err)

      activeRef.current = false
      stoppingRef.current = true

      setError(err?.message || 'Unable to start audio recorder.')
      setStatus('Error')
    }
  }

  const sendChunk = async (audioBlob, chunkNumber) => {
    const sessionId = sessionIdRef.current

    if (!sessionId) {
      console.error('DEEPSHIELD: NO SESSION ID')
      return
    }

    try {
      setStatus(`Analyzing chunk ${chunkNumber}...`)

      const data = await analyzeLiveChunk(sessionId, audioBlob)

      console.log('DEEPSHIELD LIVE RESULT:', data)

      if (!activeRef.current || stoppingRef.current) {
        return
      }

      setChunksSent((count) => count + 1)

      setResult({ ...data, chunk: chunkNumber })

      const label = data.label || data.state || 'UNKNOWN'
      const confidence = data.confidence

      if (confidence !== undefined && confidence !== null) {
        setStatus(`Detected: ${label} (${confidence}%)`)
      } else {
        setStatus(`Detected: ${label}`)
      }

    } catch (err) {
      console.error(`DEEPSHIELD: CHUNK ${chunkNumber} ANALYSIS ERROR`, err)

      if (activeRef.current && !stoppingRef.current) {
        setError(err?.message || 'Unable to analyze live audio.')
        setStatus('Listening...')
      }

      if (err.message?.includes('401') || err.message?.includes('Authentication')) {
        setAuthError('Authentication required. Please sign in again.')
        stopDetection()
      }
    }
  }

  const releaseResources = async () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }

    const recorder = recorderRef.current
    recorderRef.current = null

    if (recorder && recorder.state !== 'inactive') {
      try {
        recorder.stop()
      } catch {}
    }

    const stream = streamRef.current
    streamRef.current = null

    if (stream) {
      stream.getTracks().forEach((track) => {
        try {
          track.stop()
        } catch {}
      })
    }
  }

  const stopDetection = async () => {
    if (!activeRef.current) {
      return
    }

    console.log('DEEPSHIELD: STOPPING WEB LIVE DETECTION')

    activeRef.current = false
    stoppingRef.current = true

    setStatus('Stopping detection...')

    const sessionId = sessionIdRef.current
    sessionIdRef.current = null

    await releaseResources()

    if (sessionId) {
      try {
        await stopLiveSession(sessionId)
      } catch (err) {
        console.error('DEEPSHIELD SESSION STOP ERROR:', err)
      }
    }

    setIsDetecting(false)
    setStatus('Stopped')
  }

  const handleToggle = () => {
    if (activeRef.current) {
      stopDetection()
    } else {
      startDetection()
    }
  }

  if (authLoading) {
    return (
      <div className="live-detection">
        <div className="loading">Loading authentication...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="live-detection">
        <div className="auth-required">
          <h2>Authentication Required</h2>
          <p>{authError || 'Please sign in to use live detection'}</p>
        </div>
      </div>
    )
  }

  return ( <div className="live-detection">

  <div className="dev-banner" role="banner" aria-label="Under development notice">
    <div className="dev-banner-tape">UNDER DEVELOPMENT</div>
    <div className="dev-banner-text">
      <strong>⚠ Live Detection is Under Active Development</strong>
      <span>This feature is not yet production-ready. Microphone capture, chunking, and backend integration are implemented but may have issues. Use the <a href="#" onClick={(e) => { e.preventDefault(); onNavigate?.('analyze'); }}>Analyze Audio</a> page for stable file-based detection.</span>
    </div>
  </div>

  <div className="live-header">

    <div>

      <button
        className="back-button"
        onClick={() => {
          if (activeRef.current) {
            stopDetection()
          }

          if (onNavigate) {
            onNavigate('dashboard')
          }
        }}
      >
        ← Back to Dashboard
      </button>

      <h1>
        Live Detection
      </h1>

      <p>
        Monitor audio continuously for synthetic speech.
      </p>

    </div>

    <div className="live-status">

      <span
        className={
          `live-status-dot ${
            isDetecting
              ? 'active'
              : ''
          }`
        }
      />

      {status}

    </div>

  </div>

  <div className="live-panel">

    <div className="audio-visualizer">

      {Array.from({
        length: 32
      }).map(
        (_, index) => (
          <div
            key={index}
            className={
              `audio-bar ${
                isDetecting
                  ? 'active'
                  : ''
              }`
            }
            style={{
              animationDelay:
                `${index * 0.04}s`
            }}
          />
        )
      )}

    </div>

    <div className="live-controls">

      <button
        className={
          `live-button ${
            isDetecting
              ? 'stop-button'
              : 'start-button'
          }`
        }
        onClick={handleToggle}
      >
        {isDetecting
          ? 'Stop Detection'
          : 'Start Detection (Beta)'}
      </button>

    </div>

    {error && (

      <div className="detection-result">

        <h2>
          Capture Error
        </h2>

        <p>
          {error}
        </p>

      </div>

    )}

    {result && (

      <div className="detection-result">

        <h2>
          {result.label ||
            'Analysis Complete'}
        </h2>

        <div className="detection-confidence">

          {result.confidence !==
          undefined &&
          result.confidence !==
          null
            ? `${result.confidence}%`
            : '—'}

        </div>

        <div className="detection-state">

          {result.label ||
            result.state ||
            'Unknown'}

        </div>

        {result.chunk && (

          <p>
            Chunk {result.chunk} analyzed successfully.
          </p>

        )}

      </div>

    )}

    <div className="live-info">

      <div className="live-info-card">

        <span>
          Capture Status
        </span>

        <strong>
          {isDetecting
            ? 'ACTIVE'
            : 'IDLE'}
        </strong>

      </div>

      <div className="live-info-card">

        <span>
          Chunks Sent
        </span>

        <strong>
          {chunksSent}
        </strong>

      </div>

      <div className="live-info-card">

        <span>
          Detection Engine
        </span>

        <strong>
          {result
            ? 'PROCESSING'
            : 'READY'}
        </strong>

      </div>

    </div>

  </div>

</div>
)
}

export default LiveDetection