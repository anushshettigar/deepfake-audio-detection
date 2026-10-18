import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

if (!supabaseUrl) {
  throw new Error('VITE_SUPABASE_URL is not configured')
}

if (!supabaseAnonKey) {
  throw new Error('VITE_SUPABASE_ANON_KEY is not configured')
}

export const supabase = createClient(
  supabaseUrl,
  supabaseAnonKey
)

export async function getSession() {
  const {
    data: { session }
  } = await supabase.auth.getSession()

  return { session }
}

export async function getAccessToken() {
  const { session } = await getSession()

  return session?.access_token || null
}

export function onAuthStateChange(callback) {
  return supabase.auth.onAuthStateChange(callback)
}

export async function signIn(email, password) {
  const { data, error } =
    await supabase.auth.signInWithPassword({
      email,
      password
    })

  return { data, error }
}

export async function signUp(email, password) {
  const { data, error } =
    await supabase.auth.signUp({
      email,
      password
    })

  return { data, error }
}

export async function signOut() {
  const { error } =
    await supabase.auth.signOut()

  return { error }
}