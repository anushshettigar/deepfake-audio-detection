import { StrictMode, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { AuthProvider } from './context/AuthContext.jsx'
import { supabase } from './lib/supabase.js'

function ExtensionAuthBridge() {
  useEffect(() => {
    let active = true

    const handleMessage = async (event) => {
      if (event.source !== window) {
        return
      }

      if (
        event.origin !== window.location.origin
      ) {
        return
      }

      if (
        event.data?.source !==
          'DEEPSHIELD_EXTENSION' ||
        event.data?.type !==
          'DEEPSHIELD_AUTH_REQUEST'
      ) {
        return
      }

      try {
        const {
          data,
          error
        } = await supabase.auth.getSession()

        if (!active) {
          return
        }

        window.postMessage(
          {
            source:
              'DEEPSHIELD_WEBSITE',
            type:
              'DEEPSHIELD_AUTH_RESPONSE',
            access_token:
              error
                ? null
                : data?.session
                    ?.access_token || null,
            error:
              error?.message || null
          },
          window.location.origin
        )
      } catch (error) {
        if (!active) {
          return
        }

        window.postMessage(
          {
            source:
              'DEEPSHIELD_WEBSITE',
            type:
              'DEEPSHIELD_AUTH_RESPONSE',
            access_token:
              null,
            error:
              error?.message ||
              'Unable to retrieve authentication session.'
          },
          window.location.origin
        )
      }
    }

    window.addEventListener(
      'message',
      handleMessage
    )

    return () => {
      active = false

      window.removeEventListener(
        'message',
        handleMessage
      )
    }
  }, [])

  return null
}

createRoot(
  document.getElementById('root')
).render(
  <StrictMode>
    <AuthProvider>
      <ExtensionAuthBridge />
      <App />
    </AuthProvider>
  </StrictMode>
)