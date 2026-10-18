import { useState } from 'react'
import { useAuth } from '../context/AuthContext.jsx'
import './Login.css'

function Login({ onLogin, onSignup }) {
  const { signIn, loading: authLoading } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)

    try {
      const { error } = await signIn(email, password)

      if (error) {
        setError(error.message)
      } else {
        onLogin?.()
      }
    } catch (err) {
      setError(err.message || 'Login failed')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-header">
          <div className="login-brand">
            <div className="brand-mark">DS</div>
            <div>
              <strong>DEEPSHIELD</strong>
              <span>Audio Intelligence</span>
            </div>
          </div>

          <h1>Sign In</h1>
          <p>Enter your credentials to access the dashboard</p>
        </div>

        <form className="login-form" onSubmit={handleSubmit}>
          {error && (
            <div className="login-error">
              {error}
            </div>
          )}

          <div className="form-group">
            <label htmlFor="email">Email</label>

            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
              autoComplete="email"
              disabled={isLoading || authLoading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>

            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              autoComplete="current-password"
              disabled={isLoading || authLoading}
            />
          </div>

          <button
            type="submit"
            className="login-button"
            disabled={isLoading || authLoading}
          >
            {isLoading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <div className="login-footer">
          <p>
            No account?{' '}
            <button
              type="button"
              className="login-link-button"
              onClick={onSignup}
              disabled={isLoading || authLoading}
            >
              Sign up
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}

export default Login