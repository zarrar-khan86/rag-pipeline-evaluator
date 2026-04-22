import { useEffect, useMemo, useState } from 'react'

const pipelineOptions = [
  { label: 'RAG Fusion', value: 'rag_fusion' },
  { label: 'HyDE', value: 'hyde' },
  { label: 'CRAG', value: 'crag' },
  { label: 'Graph RAG', value: 'graph_rag' },
]

const styles = {
  page: {
    minHeight: '100vh',
    margin: 0,
    fontFamily: 'Georgia, "Times New Roman", serif',
    color: '#1f1f1f',
    background:
      'radial-gradient(circle at 20% 20%, #f9e5b7 0%, #f6efe0 35%, #e8f0ea 100%)',
    padding: '24px',
  },
  panel: {
    maxWidth: '1000px',
    margin: '0 auto',
    background: 'rgba(255,255,255,0.88)',
    border: '1px solid #d2c7ae',
    borderRadius: '16px',
    padding: '20px',
    boxShadow: '0 20px 40px rgba(40, 32, 22, 0.08)',
  },
  row: {
    display: 'flex',
    gap: '12px',
    flexWrap: 'wrap',
    marginBottom: '14px',
  },
  input: {
    flex: 1,
    minWidth: '250px',
    padding: '10px 12px',
    borderRadius: '10px',
    border: '1px solid #b7ab8f',
    fontSize: '15px',
    background: '#fffaf0',
  },
  select: {
    padding: '10px 12px',
    borderRadius: '10px',
    border: '1px solid #b7ab8f',
    background: '#fffaf0',
  },
  button: {
    padding: '10px 16px',
    borderRadius: '10px',
    border: 'none',
    background: '#265d4c',
    color: 'white',
    fontWeight: 700,
    cursor: 'pointer',
  },
  card: {
    marginTop: '14px',
    padding: '12px',
    borderRadius: '12px',
    border: '1px solid #ddd3bf',
    background: '#fffdf8',
  },
  chunk: {
    marginTop: '10px',
    padding: '10px',
    borderRadius: '8px',
    border: '1px solid #e2dbc7',
    background: '#fff',
  },
}

export default function App() {
  const [query, setQuery] = useState('Who directed Inception?')
  const [pipeline, setPipeline] = useState('rag_fusion')
  const [topK, setTopK] = useState(3)
  const [samples, setSamples] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const sampleQueries = useMemo(() => samples.map((s) => s.query), [samples])

  useEffect(() => {
    fetch('/api/samples?limit=12')
      .then((r) => r.json())
      .then((data) => setSamples(data.samples || []))
      .catch(() => setSamples([]))
  }, [])

  async function runPipeline() {
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, pipeline, top_k: topK }),
      })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.error || 'Request failed')
      }
      setResult(data)
    } catch (e) {
      setError(String(e.message || e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.page}>
      <div style={styles.panel}>
        <h1 style={{ marginTop: 0 }}>RAG in the Wild: Pipeline Explorer</h1>
        <p>
          Run any strategy on the shared global corpus index and inspect retrieved evidence,
          retrieval scores, confidence, and final answer.
        </p>

        <div style={styles.row}>
          <input
            style={styles.input}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your query"
          />
          <select style={styles.select} value={pipeline} onChange={(e) => setPipeline(e.target.value)}>
            {pipelineOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <input
            style={{ ...styles.select, width: '80px' }}
            type="number"
            min={1}
            max={10}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value || 3))}
          />
          <button style={styles.button} onClick={runPipeline} disabled={loading || !query.trim()}>
            {loading ? 'Running...' : 'Run'}
          </button>
        </div>

        <div style={styles.row}>
          <label htmlFor="sampleSelect">Sample Query:</label>
          <select
            id="sampleSelect"
            style={{ ...styles.select, minWidth: '320px' }}
            onChange={(e) => setQuery(e.target.value)}
            value=""
          >
            <option value="" disabled>
              Choose from dataset
            </option>
            {sampleQueries.map((q, idx) => (
              <option key={idx} value={q}>
                {q}
              </option>
            ))}
          </select>
        </div>

        {error ? <div style={{ color: '#9b2226', fontWeight: 700 }}>{error}</div> : null}

        {result ? (
          <div style={styles.card}>
            <h2 style={{ marginTop: 0 }}>Answer</h2>
            <div>{result.answer}</div>

            {result?.meta?.retrieval_confidence !== undefined ? (
              <p>
                <strong>CRAG retrieval confidence:</strong>{' '}
                {Number(result.meta.retrieval_confidence).toFixed(4)}
              </p>
            ) : null}

            {result.hypothetical_document ? (
              <>
                <h3>HyDE Hypothetical Document</h3>
                <div style={styles.chunk}>{result.hypothetical_document}</div>
              </>
            ) : null}

            <h3>Retrieved Chunks</h3>
            {(result.retrieved_chunks || []).map((chunk, idx) => (
              <div key={idx} style={styles.chunk}>
                <div>
                  <strong>Rank:</strong> {chunk.rank || idx + 1} | <strong>Score:</strong>{' '}
                  {Number(chunk.score || 0).toFixed(4)}
                  {chunk.fusion_score !== undefined
                    ? ` | Fusion: ${Number(chunk.fusion_score).toFixed(4)}`
                    : ''}
                  {chunk.graph_score !== undefined
                    ? ` | Graph: ${Number(chunk.graph_score).toFixed(4)}`
                    : ''}
                </div>
                <div>
                  <strong>Source:</strong> {chunk.source_title || 'Unknown source'}
                </div>
                <div>{chunk.chunk_text}</div>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  )
}
