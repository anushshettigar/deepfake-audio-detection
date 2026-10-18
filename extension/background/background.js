let captureTabId = null
let capturePageTabId = null

const DEFAULT_STATE = {
  detecting: false,
  status: "Ready",
  statusText:
    "Start detection to monitor call audio",
  result: "—",
  confidence:
    "Waiting for audio",
  engineStatus: "READY",
  chunksReceived: 0,
  lastAnalysis: null,
  sessionId: null
}

const DEEPSHIELD_FRONTEND_PATTERNS = [
  "http://localhost:5173/*",
  "http://127.0.0.1:5173/*",
  "https://deepshield.app/*",
  "https://www.deepshield.app/*"
]

chrome.runtime.onInstalled.addListener(
  async () => {
    await chrome.storage.local.set(
      DEFAULT_STATE
    )

    await chrome.storage.session.remove(
      "supabase_access_token"
    )
  }
)

chrome.runtime.onStartup.addListener(
  async () => {
    const existing =
      await chrome.storage.local.get(
        Object.keys(DEFAULT_STATE)
      )

    if (
      existing.detecting ===
      undefined
    ) {
      await chrome.storage.local.set(
        DEFAULT_STATE
      )
    }

    await chrome.storage.session.remove(
      "supabase_access_token"
    )
  }
)

chrome.runtime.onMessage.addListener(
  async (
    message,
    sender,
    sendResponse
  ) => {
    console.log(
      "DeepShield background message:",
      message
    )

    if (
      message?.type ===
      "GET_AUTH_TOKEN"
    ) {
      try {
        const token =
          await refreshAuthToken()

        if (!token) {
          sendResponse({
            status: "error",
            message:
              "No authenticated DeepShield session was found. Open the DeepShield website and sign in."
          })

          return true
        }

        sendResponse({
          status: "ok",
          access_token:
            token
        })
      } catch (error) {
        console.error(
          "DeepShield auth error:",
          error
        )

        await chrome.storage.session.remove(
          "supabase_access_token"
        )

        sendResponse({
          status: "error",
          message:
            error?.message ||
            "Unable to obtain authentication token."
        })
      }

      return true
    }

    if (
      message?.type ===
      "START_DETECTION"
    ) {
      try {
        const tabs =
          await chrome.tabs.query({
            active: true,
            currentWindow: true
          })

        if (
          !tabs ||
          tabs.length === 0 ||
          !tabs[0]?.id
        ) {
          throw new Error(
            "No active browser tab found."
          )
        }

        const tab =
          tabs[0]

        captureTabId =
          tab.id

        const accessToken =
          await refreshAuthToken()

        if (!accessToken) {
          throw new Error(
            "Please open the DeepShield website and sign in before starting live detection."
          )
        }

        await chrome.storage.session.set({
          supabase_access_token:
            accessToken
        })

        const captureUrl =
          chrome.runtime.getURL(
            "capture/capture.html"
          )

        const captureTab =
          await chrome.tabs.create({
            url: captureUrl,
            active: false
          })

        capturePageTabId =
          captureTab.id

        await waitForTabLoad(
          captureTab.id
        )

        const response =
          await chrome.tabs.sendMessage(
            captureTab.id,
            {
              type:
                "CAPTURE_START",
              targetTabId:
                captureTabId
            }
          )

        if (
          response?.status !==
          "started"
        ) {
          throw new Error(
            response?.message ||
            "Audio capture failed."
          )
        }

        await chrome.storage.local.set({
          detecting: true,
          status:
            "Listening...",
          statusText:
            "Call audio is being analyzed",
          result:
            "CAPTURING",
          confidence:
            "Waiting for analysis",
          engineStatus:
            "ACTIVE",
          chunksReceived: 0,
          lastAnalysis: null,
          sessionId:
            response.session_id ||
            null
        })

        sendResponse({
          status: "started"
        })
      } catch (error) {
        console.error(
          "DeepShield start error:",
          error
        )

        await cleanupCaptureTabs()

        await chrome.storage.local.set({
          detecting: false,
          status:
            "Capture Error",
          statusText:
            error?.message ||
            "Unable to start detection.",
          result:
            "ERROR",
          confidence:
            "Unable to analyze audio",
          engineStatus:
            "ERROR",
          sessionId:
            null
        })

        await chrome.storage.session.remove(
          "supabase_access_token"
        )

        sendResponse({
          status: "error",
          message:
            error?.message ||
            "Unable to start detection."
        })
      }

      return true
    }

    if (
      message?.type ===
      "DETECTION_RESULT"
    ) {
      try {
        const data =
          message.data || {}

        const current =
          await chrome.storage.local.get([
            "chunksReceived"
          ])

        const chunks =
          (
            current.chunksReceived ||
            0
          ) + 1

        const label =
          data.label ||
          data.state ||
          data.status ||
          "PENDING"

        const confidenceValue =
          data.confidence

        const confidenceText =
          confidenceValue !==
            undefined &&
          confidenceValue !==
            null
            ? `${confidenceValue}% confidence`
            : `Audio chunks analyzed: ${chunks}`

        await chrome.storage.local.set({
          detecting: true,
          status:
            "Listening...",
          statusText:
            "Call audio is being analyzed",
          result:
            String(label).toUpperCase(),
          confidence:
            confidenceText,
          engineStatus:
            "PROCESSING",
          chunksReceived:
            chunks,
          lastAnalysis:
            data,
          sessionId:
            data.session_id ||
            null
        })

        sendResponse({
          status: "stored"
        })
      } catch (error) {
        console.error(
          "Failed to store detection result:",
          error
        )

        sendResponse({
          status: "error",
          message:
            error?.message ||
            "Unable to store detection result."
        })
      }

      return true
    }

    if (
      message?.type ===
      "DETECTION_ERROR"
    ) {
      console.error(
        "DeepShield detection error:",
        message.message
      )

      await chrome.storage.local.set({
        detecting: false,
        status:
          "Capture Error",
        statusText:
          message.message ||
          "Backend analysis failed.",
        result:
          "ERROR",
        confidence:
          "Unable to analyze audio",
        engineStatus:
          "ERROR"
      })

      sendResponse({
        status: "stored"
      })

      return true
    }

    if (
      message?.type ===
      "STOP_DETECTION"
    ) {
      try {
        await cleanupCaptureTabs()

        await chrome.storage.local.set({
          detecting: false,
          status:
            "Stopped",
          statusText:
            "Detection stopped",
          engineStatus:
            "READY",
          sessionId:
            null
        })

        await chrome.storage.session.remove(
          "supabase_access_token"
        )

        sendResponse({
          status: "stopped"
        })
      } catch (error) {
        console.error(
          "DeepShield stop error:",
          error
        )

        sendResponse({
          status: "error",
          message:
            error?.message ||
            "Unable to stop detection."
        })
      }

      return true
    }
  }
)

async function refreshAuthToken() {
  const tabs =
    await chrome.tabs.query({
      url:
        DEEPSHIELD_FRONTEND_PATTERNS
    })

  if (
    !tabs ||
    tabs.length === 0
  ) {
    throw new Error(
      "Open the DeepShield website in a browser tab and sign in before starting live detection."
    )
  }

  const preferredTab =
    tabs.find(
      tab =>
        tab.active &&
        tab.id
    ) ||
    tabs.find(
      tab =>
        tab.id
    )

  if (!preferredTab?.id) {
    throw new Error(
      "No usable DeepShield website tab was found."
    )
  }

  let response

  try {
    response =
      await chrome.tabs.sendMessage(
        preferredTab.id,
        {
          type:
            "DEEPSHIELD_AUTH_REQUEST"
        }
      )
  } catch (error) {
    console.error(
      "DeepShield auth bridge error:",
      error
    )

    throw new Error(
      "The DeepShield website tab is open, but its authentication bridge is unavailable. Reload the DeepShield website and try again."
    )
  }

  const token =
    response?.access_token

  if (
    typeof token !== "string" ||
    !token.trim()
  ) {
    await chrome.storage.session.remove(
      "supabase_access_token"
    )

    return null
  }

  const cleanToken =
    token.trim()

  await chrome.storage.session.set({
    supabase_access_token:
      cleanToken
  })

  return cleanToken
}

async function cleanupCaptureTabs() {
  if (
    capturePageTabId !==
    null
  ) {
    try {
      await chrome.tabs.sendMessage(
        capturePageTabId,
        {
          type:
            "CAPTURE_STOP"
        }
      )
    } catch (error) {
      console.warn(
        "Could not stop capture page:",
        error
      )
    }

    try {
      await chrome.tabs.remove(
        capturePageTabId
      )
    } catch (error) {
      console.warn(
        "Could not close capture page:",
        error
      )
    }
  }

  capturePageTabId =
    null

  captureTabId =
    null
}

function waitForTabLoad(
  tabId
) {
  return new Promise(
    resolve => {
      let settled = false

      const finish = () => {
        if (settled) {
          return
        }

        settled = true

        chrome.tabs.onUpdated.removeListener(
          listener
        )

        resolve()
      }

      const listener =
        (
          updatedTabId,
          changeInfo
        ) => {
          if (
            updatedTabId ===
              tabId &&
            changeInfo.status ===
              "complete"
          ) {
            finish()
          }
        }

      chrome.tabs.get(
        tabId,
        tab => {
          if (
            chrome.runtime.lastError
          ) {
            finish()
            return
          }

          if (
            tab.status ===
            "complete"
          ) {
            finish()
            return
          }

          chrome.tabs.onUpdated.addListener(
            listener
          )
        }
      )

      setTimeout(
        finish,
        15000
      )
    }
  )
}