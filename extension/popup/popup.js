const startButton =
  document.getElementById(
    "startButton"
  )

const status =
  document.getElementById(
    "status"
  )

const statusText =
  document.getElementById(
    "statusText"
  )

const result =
  document.getElementById(
    "result"
  )

const confidence =
  document.getElementById(
    "confidence"
  )

const engineStatus =
  document.getElementById(
    "engineStatus"
  )


let detecting = false


const DEFAULT_STATE = {

  detecting: false,

  status:
    "Ready",

  statusText:
    "Start detection to monitor call audio",

  result:
    "—",

  confidence:
    "Waiting for audio",

  engineStatus:
    "READY",

  chunksReceived:
    0

}


function updateUI(
  state
) {

  const data = {
    ...DEFAULT_STATE,
    ...state
  }


  detecting =
    Boolean(
      data.detecting
    )


  status.textContent =
    data.status


  statusText.textContent =
    data.statusText


  result.textContent =
    data.result


  confidence.textContent =
    data.confidence


  engineStatus.textContent =
    data.engineStatus


  startButton.textContent =
    detecting
      ? "Stop Detection"
      : "Start Detection"


  if (detecting) {

    startButton.classList.add(
      "active"
    )

  } else {

    startButton.classList.remove(
      "active"
    )

  }

}


async function loadState() {

  try {

    const state =
      await chrome.storage.local.get(
        Object.keys(
          DEFAULT_STATE
        )
      )


    updateUI(
      state
    )

  } catch (error) {

    console.error(
      "Unable to load DeepShield state:",
      error
    )

  }

}


function setErrorState(
  message
) {

  updateUI({

    detecting: false,

    status:
      "Capture Error",

    statusText:
      message,

    result:
      "ERROR",

    confidence:
      "Unable to analyze audio",

    engineStatus:
      "ERROR"

  })

}


startButton.addEventListener(
  "click",
  async () => {

    startButton.disabled =
      true


    try {

      if (!detecting) {

        updateUI({

          detecting: false,

          status:
            "Starting...",

          statusText:
            "Requesting tab audio access",

          result:
            "STARTING",

          confidence:
            "Waiting for audio",

          engineStatus:
            "STARTING"

        })


        const response =
          await chrome.runtime.sendMessage({

            type:
              "START_DETECTION"

          })


        console.log(
          "Start response:",
          response
        )


        if (
          !response ||
          response.status !==
            "started"
        ) {

          throw new Error(
            response?.message ||
            "Unable to start audio capture."
          )

        }


        await loadState()

      } else {

        await chrome.runtime.sendMessage({

          type:
            "STOP_DETECTION"

        })


        await loadState()

      }

    } catch (error) {

      console.error(
        "DeepShield popup error:",
        error
      )


      setErrorState(
        error.message ||
        "Unable to control detection."
      )

    } finally {

      startButton.disabled =
        false

    }

  }
)


chrome.storage.onChanged.addListener(
  (
    changes,
    areaName
  ) => {

    if (
      areaName !==
      "local"
    ) {

      return

    }


    loadState()

  }
)


document.addEventListener(
  "DOMContentLoaded",
  loadState
)


loadState()