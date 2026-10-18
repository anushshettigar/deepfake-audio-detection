let captureStream = null
let audioContext = null
let mediaRecorder = null
let currentSessionId = null
let targetTabId = null

const DEFAULT_BACKEND_URL =
  "http://127.0.0.1:8000"

const CHUNK_DURATION = 3000

async function getBackendUrl() {
  try {
    const result =
      await chrome.storage.local.get([
        "backend_url"
      ])

    const storedUrl =
      typeof result.backend_url ===
        "string"
        ? result.backend_url
            .trim()
            .replace(/\/+$/, "")
        : ""

    if (
      storedUrl ===
        "http://127.0.0.1:8000" ||
      storedUrl ===
        "http://localhost:8000" ||
      storedUrl ===
        "https://api.deepshield.app"
    ) {
      return storedUrl
    }

    return DEFAULT_BACKEND_URL
  } catch {
    return DEFAULT_BACKEND_URL
  }
}

async function getAccessToken() {
  const result =
    await chrome.storage.session.get([
      "supabase_access_token"
    ])

  const token =
    result.supabase_access_token

  if (
    typeof token !== "string" ||
    !token.trim()
  ) {
    throw new Error(
      "DeepShield authentication session is missing. Please sign in to DeepShield and try again."
    )
  }

  return token.trim()
}

let recorderChunks = []
let chunkTimer = null
let stopping = false
let processingChunk = false

chrome.runtime.onMessage.addListener(
  async (
    message,
    sender,
    sendResponse
  ) => {
    console.log(
      "DeepShield capture message:",
      message
    )

    if (
      message.type ===
      "CAPTURE_START"
    ) {
      try {
        targetTabId =
          message.targetTabId ||
          null

        const sessionId =
          await startCapture()

        sendResponse({
          status: "started",
          session_id:
            sessionId
        })
      } catch (error) {
        console.error(
          "DeepShield capture start error:",
          error
        )

        await stopCapture()

        sendResponse({
          status: "error",
          message:
            error?.message ||
            "Audio capture failed."
        })
      }

      return true
    }

    if (
      message.type ===
      "CAPTURE_STOP"
    ) {
      try {
        await stopCapture()

        sendResponse({
          status: "stopped"
        })
      } catch (error) {
        console.error(
          "DeepShield capture stop error:",
          error
        )

        sendResponse({
          status: "error",
          message:
            error?.message ||
            "Unable to stop capture."
        })
      }

      return true
    }
  }
)

function getMimeType() {
  const types = [
    "audio/webm;codecs=opus",
    "audio/webm"
  ]

  return types.find(
    type =>
      MediaRecorder.isTypeSupported(
        type
      )
  )
}

async function startCapture() {
  if (mediaRecorder) {
    return currentSessionId
  }

  if (!targetTabId) {
    throw new Error(
      "No target browser tab was provided."
    )
  }

  stopping = false
  processingChunk = false
  recorderChunks = []

  const backendUrl =
    await getBackendUrl()

  const accessToken =
    await getAccessToken()

  console.log(
    "DeepShield backend:",
    backendUrl
  )

  console.log(
    "DeepShield authentication token available:",
    Boolean(accessToken)
  )

  const sessionResponse =
    await fetch(
      `${backendUrl}/api/live-session/start`,
      {
        method: "POST",
        headers: {
          Authorization:
            `Bearer ${accessToken}`,
          Accept:
            "application/json"
        }
      }
    )

  const sessionText =
    await sessionResponse.text()

  if (
    sessionResponse.status === 401
  ) {
    await chrome.storage.session.remove(
      "supabase_access_token"
    )

    throw new Error(
      "DeepShield authentication was rejected by the backend. Please sign in to DeepShield again and restart live detection."
    )
  }

  if (!sessionResponse.ok) {
    throw new Error(
      sessionText ||
      `Unable to create live detection session. HTTP ${sessionResponse.status}`
    )
  }

  let sessionData

  try {
    sessionData =
      JSON.parse(
        sessionText
      )
  } catch {
    throw new Error(
      "Backend returned an invalid live-session response."
    )
  }

  if (
    !sessionData.session_id
  ) {
    throw new Error(
      "Backend did not return a live session ID."
    )
  }

  currentSessionId =
    sessionData.session_id

  console.log(
    "DeepShield session:",
    currentSessionId
  )

  console.log(
    "DeepShield: requesting tab audio stream:",
    targetTabId
  )

  const streamId =
    await chrome.tabCapture.getMediaStreamId({
      targetTabId
    })

  if (!streamId) {
    throw new Error(
      "Unable to obtain the browser tab audio stream."
    )
  }

  if (
    !navigator.mediaDevices ||
    !navigator.mediaDevices.getUserMedia
  ) {
    throw new Error(
      "Audio capture is not available in this extension."
    )
  }

  captureStream =
    await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource:
            "tab",
          chromeMediaSourceId:
            streamId
        }
      },
      video: false
    })

  if (!captureStream) {
    throw new Error(
      "Unable to create the tab audio stream."
    )
  }

  console.log(
    "DeepShield: tab audio captured"
  )

  audioContext =
    new AudioContext()

  if (
    audioContext.state ===
    "suspended"
  ) {
    await audioContext.resume()
  }

  const source =
    audioContext
      .createMediaStreamSource(
        captureStream
      )

  const destination =
    audioContext
      .createMediaStreamDestination()

  source.connect(
    destination
  )

  await startRecorder(
    destination.stream
  )

  console.log(
    "DeepShield: live recording started"
  )

  return currentSessionId
}

async function startRecorder(
  stream
) {
  if (stopping) {
    return
  }

  const mimeType =
    getMimeType()

  if (!mimeType) {
    throw new Error(
      "No compatible audio recording format is supported."
    )
  }

  console.log(
    "DeepShield audio format:",
    mimeType
  )

  recorderChunks = []

  mediaRecorder =
    new MediaRecorder(
      stream,
      {
        mimeType
      }
    )

  mediaRecorder.onstart =
    () => {
      console.log(
        "DeepShield: MediaRecorder started"
      )
    }

  mediaRecorder.ondataavailable =
    event => {
      if (
        event.data &&
        event.data.size > 0
      ) {
        recorderChunks.push(
          event.data
        )

        console.log(
          "DeepShield recorder data:",
          event.data.size,
          "bytes"
        )
      }
    }

  mediaRecorder.onstop =
    async () => {
      console.log(
        "DeepShield: MediaRecorder stopped"
      )

      if (stopping) {
        recorderChunks = []
        mediaRecorder = null
        return
      }

      const chunks =
        recorderChunks

      recorderChunks = []
      mediaRecorder = null

      if (!chunks.length) {
        console.warn(
          "DeepShield: no audio data collected"
        )

        await restartRecorder()
        return
      }

      const completeBlob =
        new Blob(
          chunks,
          {
            type: mimeType
          }
        )

      console.log(
        "DeepShield complete recording:",
        completeBlob.size,
        "bytes"
      )

      if (
        completeBlob.size < 1000
      ) {
        console.warn(
          "DeepShield: recording is too small, skipping"
        )

        await restartRecorder()
        return
      }

      processingChunk = true

      try {
        await sendAudioChunk(
          completeBlob,
          mimeType
        )
      } finally {
        processingChunk = false
      }

      if (!stopping) {
        await restartRecorder()
      }
    }

  mediaRecorder.onerror =
    event => {
      console.error(
        "DeepShield recorder error:",
        event.error
      )

      chrome.runtime.sendMessage({
        type:
          "DETECTION_ERROR",
        message:
          event.error?.message ||
          "Audio recorder encountered an error."
      })
    }

  mediaRecorder.start()

  console.log(
    "DeepShield: recorder started"
  )

  if (chunkTimer) {
    clearTimeout(
      chunkTimer
    )
  }

  chunkTimer =
    setTimeout(
      () => {
        if (
          mediaRecorder &&
          mediaRecorder.state !==
            "inactive"
        ) {
          console.log(
            "DeepShield: 3 seconds reached, stopping recorder"
          )

          mediaRecorder.stop()
        }
      },
      CHUNK_DURATION
    )
}

async function restartRecorder() {
  if (stopping) {
    return
  }

  if (!captureStream) {
    return
  }

  if (!audioContext) {
    return
  }

  const destinationStream =
    audioContext
      .createMediaStreamDestination()

  const source =
    audioContext
      .createMediaStreamSource(
        captureStream
      )

  source.connect(
    destinationStream
  )

  await startRecorder(
    destinationStream.stream
  )
}

async function sendAudioChunk(
  audioBlob,
  mimeType
) {
  if (!currentSessionId) {
    console.error(
      "DeepShield: no active session."
    )

    return
  }

  try {
    const backendUrl =
      await getBackendUrl()

    const accessToken =
      await getAccessToken()

    const extension =
      mimeType.includes("ogg")
        ? "ogg"
        : "webm"

    const formData =
      new FormData()

    formData.append(
      "file",
      audioBlob,
      `live-${Date.now()}.${extension}`
    )

    formData.append(
      "session_id",
      currentSessionId
    )

    console.log(
      "DeepShield: uploading complete authenticated audio"
    )

    const response =
      await fetch(
        `${backendUrl}/api/live-analyze`,
        {
          method: "POST",
          headers: {
            Authorization:
              `Bearer ${accessToken}`,
            Accept:
              "application/json"
          },
          body:
            formData
        }
      )

    const responseText =
      await response.text()

    if (
      response.status === 401
    ) {
      await chrome.storage.session.remove(
        "supabase_access_token"
      )

      throw new Error(
        "DeepShield authentication expired or was rejected. Please sign in again."
      )
    }

    if (!response.ok) {
      throw new Error(
        responseText ||
        `Live audio analysis failed. HTTP ${response.status}`
      )
    }

    let data

    try {
      data =
        JSON.parse(
          responseText
        )
    } catch {
      throw new Error(
        "Backend returned an invalid JSON response."
      )
    }

    console.log(
      "DeepShield live analysis:",
      data
    )

    chrome.runtime.sendMessage({
      type:
        "DETECTION_RESULT",
      data: {
        status:
          data.status ||
          "audio_received",
        state:
          data.state ||
          "COMPLETED",
        label:
          data.label ||
          null,
        confidence:
          data.confidence ??
          null,
        class_id:
          data.class_id ??
          null,
        fake_probability:
          data.fake_probability ??
          null,
        real_probability:
          data.real_probability ??
          null,
        session_id:
          data.session_id ||
          currentSessionId,
        chunk:
          data.chunk ??
          null,
        message:
          data.message ||
          "Live audio chunk analyzed successfully."
      }
    })
  } catch (error) {
    console.error(
      "DeepShield backend error:",
      error
    )

    chrome.runtime.sendMessage({
      type:
        "DETECTION_ERROR",
      message:
        error?.message ||
        "Unable to analyze live audio."
    })
  }
}

async function stopCapture() {
  stopping = true

  if (chunkTimer) {
    clearTimeout(
      chunkTimer
    )

    chunkTimer = null
  }

  const sessionId =
    currentSessionId

  console.log(
    "DeepShield: stopping capture"
  )

  if (mediaRecorder) {
    try {
      if (
        mediaRecorder.state !==
        "inactive"
      ) {
        mediaRecorder.stop()
      }
    } catch (error) {
      console.warn(
        "DeepShield recorder stop error:",
        error
      )
    }

    mediaRecorder = null
  }

  recorderChunks = []

  if (captureStream) {
    captureStream
      .getTracks()
      .forEach(
        track => {
          try {
            track.stop()
          } catch (error) {
            console.warn(
              "DeepShield track stop error:",
              error
            )
          }
        }
      )

    captureStream = null
  }

  if (audioContext) {
    try {
      await audioContext.close()
    } catch (error) {
      console.warn(
        "DeepShield audio context close error:",
        error
      )
    }

    audioContext = null
  }

  if (sessionId) {
    try {
      const backendUrl =
        await getBackendUrl()

      const accessToken =
        await getAccessToken()

      const response =
        await fetch(
          `${backendUrl}/api/live-session/stop?session_id=${encodeURIComponent(
            sessionId
          )}`,
          {
            method: "POST",
            headers: {
              Authorization:
                `Bearer ${accessToken}`,
              Accept:
                "application/json"
            }
          }
        )

      if (response.ok) {
        const data =
          await response.json()

        console.log(
          "DeepShield session stopped:",
          data
        )
      } else {
        console.warn(
          "DeepShield session stop returned:",
          response.status
        )
      }
    } catch (error) {
      console.error(
        "DeepShield session close error:",
        error
      )
    }
  }

  currentSessionId = null
  targetTabId = null
  processingChunk = false

  console.log(
    "DeepShield: capture stopped"
  )
}