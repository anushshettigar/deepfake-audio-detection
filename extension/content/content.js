chrome.runtime.onMessage.addListener(
  (
    message,
    sender,
    sendResponse
  ) => {
    if (
      message?.type !==
      "DEEPSHIELD_AUTH_REQUEST"
    ) {
      return
    }

    try {
      const accessToken =
        getSupabaseAccessToken()

      if (!accessToken) {
        sendResponse({
          access_token: null,
          error:
            "No Supabase access token found."
        })

        return true
      }

      sendResponse({
        access_token:
          accessToken
      })
    } catch (error) {
      console.error(
        "DeepShield content auth error:",
        error
      )

      sendResponse({
        access_token: null,
        error:
          error?.message ||
          "Unable to read Supabase session."
      })
    }

    return true
  }
)

function getSupabaseAccessToken() {
  const storageKeys = []

  for (
    let index = 0;
    index < localStorage.length;
    index += 1
  ) {
    const key =
      localStorage.key(index)

    if (
      typeof key === "string" &&
      key.startsWith("sb-") &&
      key.endsWith("-auth-token")
    ) {
      storageKeys.push(key)
    }
  }

  if (
    storageKeys.length === 0
  ) {
    return null
  }

  for (
    const key of storageKeys
  ) {
    const raw =
      localStorage.getItem(key)

    if (!raw) {
      continue
    }

    try {
      const session =
        JSON.parse(raw)

      const accessToken =
        session?.access_token

      if (
        typeof accessToken ===
          "string" &&
        accessToken.trim()
      ) {
        return accessToken.trim()
      }

      const nestedAccessToken =
        session?.currentSession
          ?.access_token

      if (
        typeof nestedAccessToken ===
          "string" &&
        nestedAccessToken.trim()
      ) {
        return nestedAccessToken.trim()
      }
    } catch (error) {
      console.warn(
        "Unable to parse Supabase auth storage:",
        key,
        error
      )
    }
  }

  return null
}