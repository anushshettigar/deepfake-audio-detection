import { createContext, useContext, useEffect, useState } from 'react'
import { supabase, getSession, onAuthStateChange } from '../lib/supabase.js'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true

    const initializeSession = async () => {
      const { session: initialSession, error } = await getSession()

      if (!mounted) {
        return
      }

      if (error) {
        console.error('Failed to initialize authentication session:', error)
        setSession(null)
      } else {
        setSession(initialSession)
      }

      setLoading(false)
    }

    initializeSession()

    const {
      data: { subscription }
    } = onAuthStateChange((event, nextSession) => {
      if (!mounted) {
        return
      }

      setSession(nextSession)
      setLoading(false)
    })

    return () => {
      mounted = false
      subscription.unsubscribe()
    }
  }, [])

  const signIn = async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password
    })

    if (!error && data?.session) {
      setSession(data.session)
    }

    return { data, error }
  }

  const signUp = async (email, password) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password
    })

    if (!error && data?.session) {
      setSession(data.session)
    }

    return { data, error }
  }

  const signOut = async () => {
    const { error } = await supabase.auth.signOut()

    if (!error) {
      setSession(null)
    }

    return { error }
  }

  const value = {
    session,
    user: session?.user || null,
    loading,
    signIn,
    signUp,
    signOut,
    isAuthenticated: !!session
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)

  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }

  return context
}