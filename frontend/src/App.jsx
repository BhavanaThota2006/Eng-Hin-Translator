import { useState } from 'react'

export default function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState('')
  const [loading, setLoading] = useState(false)

  const handleTranslate = async () => {
    if (!text.trim()) return
    setLoading(true)
    setResult('')
    try {
      const res = await fetch('http://localhost:8000/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      const data = await res.json()
      setResult(data.translation ?? data.translations?.[0] ?? 'No result')
    } catch (err) {
      console.error(err)
      setResult('Error: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 100, fontFamily: 'system-ui, sans-serif' }} className='root-style'>
      <h1>English → Hindi Translator</h1>
      <textarea
        placeholder="Type English text..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={6}
        style={{ width: '100%', padding: 8, fontSize: 16 }}
      />
      <div style={{ marginTop: 8 }}>
        <button onClick={handleTranslate} disabled={loading || !text.trim()}>
          {loading ? 'Translating...' : 'Translate'}
        </button>
        <button onClick={() => { setText(''); setResult('') }} style={{ marginLeft: 8 }}>
          Clear
        </button>
      </div>

      <div style={{ marginTop: 20 }}>
        <h3>Hindi Translation</h3>
        <div style={{ minHeight: 60, padding: 8, border: '1px solid #ddd' }}>{result}</div>
      </div>
    </div>
  )
}
