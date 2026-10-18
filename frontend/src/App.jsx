import { useEffect, useState } from 'react'
import { useAuth } from './context/AuthContext.jsx'
import Intro from './Intro'
import Dashboard from './Dashboard'
import AnalyzeAudio from './AnalyzeAudio'
import LiveDetection from './LiveDetection'
import AnalysisHistory from './AnalysisHistory'
import Dataset from './Dataset'
import Settings from './Settings'
import Info from './Info'
import Login from './components/Login.jsx'
import Signup from './components/Signup.jsx'

function App() {
  const { isAuthenticated, loading: authLoading } = useAuth()
  const [showIntro, setShowIntro] = useState(true)
  const [page, setPage] = useState('dashboard')

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowIntro(false)
    }, 5500)

    return () => clearTimeout(timer)
  }, [])

  if (showIntro) {
    return <Intro />
  }

  if (authLoading) {
    return <div className="app-loading" />
  }

  if (!isAuthenticated) {
    if (page === 'signup') {
      return (
        <Signup
          onLogin={() => setPage('dashboard')}
        />
      )
    }

    return (
      <Login
        onLogin={() => setPage('dashboard')}
        onSignup={() => setPage('signup')}
      />
    )
  }

  if (page === 'dashboard') {
    return (
      <Dashboard
        onNavigate={setPage}
      />
    )
  }

  if (page === 'analyze') {
    return (
      <AnalyzeAudio
        onNavigate={setPage}
      />
    )
  }

  if (page === 'live') {
    return (
      <LiveDetection
        onNavigate={setPage}
      />
    )
  }

  if (page === 'history') {
    return (
      <AnalysisHistory
        onNavigate={setPage}
      />
    )
  }

  if (page === 'dataset') {
    return (
      <Dataset
        onNavigate={setPage}
      />
    )
  }

  if (page === 'settings') {
    return (
      <Settings
        onNavigate={setPage}
      />
    )
  }

  if (page === 'info') {
    return (
      <Info
        onNavigate={setPage}
      />
    )
  }

  return (
    <Dashboard
      onNavigate={setPage}
    />
  )
}

export default App