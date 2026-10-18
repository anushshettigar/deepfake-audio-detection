import { useAuth } from './context/AuthContext.jsx'
import './Settings.css'

function Settings({ onNavigate }) {
  const { user, signOut, isAuthenticated, loading: authLoading } = useAuth()

  if (authLoading) {
    return (
      <div className="settings-page">
        <div className="loading">Loading authentication...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="settings-page">
        <div className="auth-required">
          <h2>Authentication Required</h2>
          <p>Please sign in to access settings</p>
        </div>
      </div>
    )
  }

  const handleSignOut = async () => {
    await signOut()
  }

  return (
    <div className="settings-page">

      <button
        className="back-button"
        onClick={() => onNavigate('dashboard')}
      >
        ← Back to Dashboard
      </button>

      <div className="settings-header">
        <div>
          <span className="settings-eyebrow">
            SYSTEM CONFIGURATION
          </span>

          <h1>Settings</h1>

          <p>
            Configure your DEEPSHIELD detection environment.
          </p>
        </div>
      </div>

      <div className="settings-grid">

        <section className="settings-card">
          <div className="settings-card-header">
            <div>
              <span className="settings-label">
                DETECTION ENGINE
              </span>

              <h2>Detection Settings</h2>
            </div>

            <span className="settings-status">
              ONLINE
            </span>
          </div>

          <div className="setting-row">
            <div>
              <strong>Detection Engine</strong>

              <span>
                DEEPSHIELD neural detection engine
              </span>
            </div>

            <span className="setting-value">
              ONLINE
            </span>
          </div>

          <div className="setting-row">
            <div>
              <strong>Model Status</strong>

              <span>
                Model evaluation and training status
              </span>
            </div>

            <span className="setting-value pending">
              PENDING
            </span>
          </div>

          <div className="setting-row">
            <div>
              <strong>Audio Sample Rate</strong>

              <span>
                Processing sample rate used by the model
              </span>
            </div>

            <span className="setting-value">
              16 kHz
            </span>
          </div>

        </section>

        <section className="settings-card">
          <div className="settings-card-header">
            <div>
              <span className="settings-label">
                INFORMATION
              </span>

              <h2>About DeepShield</h2>
            </div>
          </div>

          <div className="setting-row">
            <div>
              <strong>Project Information</strong>

              <span>
                Learn about DeepShield, how to use the web app, install the extension, and meet the team
              </span>
            </div>

            <button
              className="setting-action"
              onClick={() => onNavigate('info')}
            >
              View Info
            </button>
          </div>

        </section>

        <section className="settings-card">
          <div className="settings-card-header">
            <div>
              <span className="settings-label">
                SECURITY
              </span>

              <h2>Security</h2>
            </div>
          </div>

          <div className="setting-row">
            <div>
              <strong>Secure Session</strong>

              <span>
                Current administrator session
              </span>
            </div>

            <span className="setting-value">
              ACTIVE
            </span>
          </div>

          <div className="setting-row">
            <div>
              <strong>Audio Storage</strong>

              <span>
                Uploaded recordings are stored securely
              </span>
            </div>

            <span className="setting-value">
              ENABLED
            </span>
          </div>

        </section>

        <section className="settings-card">
          <div className="settings-card-header">
            <div>
              <span className="settings-label">
                ACCOUNT
              </span>

              <h2>User Account</h2>
            </div>
          </div>

          <div className="setting-row">
            <div>
              <strong>Signed in as</strong>

              <span>
                {user?.email || 'Unknown'}
              </span>
            </div>

            <span className="setting-value">
              {user?.email || '—'}
            </span>
          </div>

          <div className="setting-row">
            <div>
              <strong>Sign Out</strong>

              <span>
                End your current session
              </span>
            </div>

            <button
              className="setting-action danger"
              onClick={handleSignOut}
            >
              Sign Out
            </button>
          </div>

        </section>

      </div>

    </div>
  )
}

export default Settings