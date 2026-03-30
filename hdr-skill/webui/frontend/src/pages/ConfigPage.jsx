import { useState, useEffect } from 'react'
import axios from 'axios'

export default function ConfigPage() {
  const [config, setConfig] = useState({
    openrouter_api_key: '',
    openrouter_model: ''
  })
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    loadConfig()
  }, [])

  const loadConfig = async () => {
    try {
      const res = await axios.get('/api/config')
      setConfig(res.data)
    } catch (err) {
      setError('Failed to load config')
    }
  }

  const saveConfig = async (e) => {
    e.preventDefault()
    setMessage('')
    setError('')
    try {
      await axios.post('/api/config', config)
      setMessage('Config saved successfully!')
    } catch (err) {
      setError('Failed to save config: ' + err.response?.data?.error || err.message)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-6">Configuration</h2>

      {message && (
        <div className="mb-4 p-4 bg-green-50 border border-green-200 text-green-700 rounded">
          {message}
        </div>
      )}

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 text-red-700 rounded">
          {error}
        </div>
      )}

      <form onSubmit={saveConfig} className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            OpenRouter API Key
          </label>
          <input
            type="password"
            value={config.openrouter_api_key || ''}
            onChange={(e) => setConfig({ ...config, openrouter_api_key: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="sk-..."
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            OpenRouter Model
          </label>
          <input
            type="text"
            value={config.openrouter_model || ''}
            onChange={(e) => setConfig({ ...config, openrouter_model: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="anthropic/claude-3-opus"
          />
          <p className="mt-1 text-sm text-gray-500">
            e.g. anthropic/claude-3-opus, openai/gpt-4o, etc.
          </p>
        </div>

        <div>
          <button
            type="submit"
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Save Configuration
          </button>
        </div>
      </form>
    </div>
  )
}
