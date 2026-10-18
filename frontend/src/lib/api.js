import { getAccessToken } from './supabase.js'

const API_BASE = ''

let unauthorizedHandler = null

export function setUnauthorizedHandler(handler) {
  unauthorizedHandler = handler
}

async function authFetch(url, options = {}) {
  const token = await getAccessToken()
  
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers
  }
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  
  // Don't set Content-Type for FormData - let browser set it with boundary
  if (options.body instanceof FormData) {
    delete headers['Content-Type']
  }
  
  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers
  })
  
  // Handle 401 - call registered handler instead of hard redirect
  if (response.status === 401) {
    if (unauthorizedHandler) {
      unauthorizedHandler()
    }
    throw new Error('Authentication required')
  }
  
  return response
}

export const api = {
  get: (url) => authFetch(url, { method: 'GET' }),
  
  post: (url, data) => {
    const isFormData = data instanceof FormData
    return authFetch(url, {
      method: 'POST',
      body: isFormData ? data : JSON.stringify(data)
    })
  },
  
  delete: (url) => authFetch(url, { method: 'DELETE' })
}

export async function fetchStats() {
  const response = await api.get('/api/stats')
  if (!response.ok) throw new Error('Failed to fetch stats')
  return response.json()
}

export async function fetchHistory(limit = 100) {
  const response = await api.get(`/api/history?limit=${limit}`)
  if (!response.ok) throw new Error('Failed to fetch history')
  return response.json()
}

export async function analyzeAudio(file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await api.post('/api/analyze', formData)
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Audio analysis failed')
  }
  return response.json()
}

export async function startLiveSession() {
  const response = await api.post('/api/live-session/start', {})
  if (!response.ok) throw new Error('Failed to start live session')
  return response.json()
}

export async function analyzeLiveChunk(sessionId, file) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('session_id', sessionId)
  const response = await api.post('/api/live-analyze', formData)
  if (!response.ok) throw new Error('Failed to analyze live chunk')
  return response.json()
}

export async function stopLiveSession(sessionId) {
  const response = await api.post(`/api/live-session/stop?session_id=${sessionId}`, {})
  if (!response.ok) throw new Error('Failed to stop live session')
  return response.json()
}

export async function fetchDataset() {
  const response = await api.get('/api/dataset')
  if (!response.ok) throw new Error('Failed to fetch dataset')
  return response.json()
}

export async function fetchTrainingDataset() {
  const response = await api.get('/api/training-dataset')
  if (!response.ok) throw new Error('Failed to fetch training dataset')
  return response.json()
}

export async function clearHistory() {
  const response = await api.delete('/api/history')
  if (!response.ok) throw new Error('Failed to clear history')
  return response.json()
}