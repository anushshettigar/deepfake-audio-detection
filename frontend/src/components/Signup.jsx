import { useState } from 'react'
import { useAuth } from '../context/AuthContext.jsx'
import './Login.css'

function Signup({ onLogin }) {
  const { signUp, loading: authLoading } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }

    setIsLoading(true)

    try {
      const { error } = await signUp(email, password)

      if (error) {
        setError(error.message)
      } else {
        onLogin?.()
      }
    } catch (err) {
      setError(err.message || 'Signup failed')
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

          <h1>Create Account</h1>
          <p>Enter your details to create an account</p>
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
              autoComplete="new-password"
              disabled={isLoading || authLoading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="confirmPassword">
              Confirm Password
            </label>

            <input
              id="confirmPassword"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="••••••••"
              required
              autoComplete="new-password"
              disabled={isLoading || authLoading}
            />
          </div>

          <button
            type="submit"
            className="login-button"
            disabled={isLoading || authLoading}
          >
            {isLoading
              ? 'Creating account...'
              : 'Create Account'}
          </button>
        </form>

        <div className="login-footer">
          <p>
            Already have an account?{' '}
            <button
              type="button"
              className="login-link-button"
              onClick={onLogin}
              disabled={isLoading || authLoading}
            >
              Sign in
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}

export default Signup