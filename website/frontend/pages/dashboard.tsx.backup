import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import { useRouter } from 'next/router'
import Head from 'next/head'
import { Container, Row, Col, Card, Button, Form, Alert, Spinner } from 'react-bootstrap'
import { 
  FaUpload, FaChartLine, FaCogs, FaSyncAlt, FaExclamationTriangle, 
  FaTachometerAlt, FaOilCan, FaSignOutAlt 
} from 'react-icons/fa'
import Plot from 'react-plotly.js'
import toast from 'react-hot-toast'
import io from 'socket.io-client'

interface ForecastData {
  date: string
  forecast_value: number
  confidence_lower: number
  confidence_upper: number
  day_ahead: number
}

interface SystemMetrics {
  cpu_usage: number
  memory_usage: number
  active_forecasts: number
  status: string
}

export default function Dashboard() {
  const { data: session, status } = useSession()
  const router = useRouter()
  const [socket, setSocket] = useState<any>(null)
  const [forecastData, setForecastData] = useState<ForecastData[]>([])
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/')
      return
    }

    // Initialize socket connection
    const newSocket = io(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000')
    setSocket(newSocket)

    // Socket event listeners
    newSocket.on('connect', () => {
      console.log('Connected to server')
    })

    newSocket.on('metrics_update', (data) => {
      setSystemMetrics(data)
    })

    newSocket.on('forecasts_update', (data) => {
      console.log('Forecasts updated:', data)
    })

    newSocket.on('error', (data) => {
      toast.error(`Error: ${data.message}`)
    })

    // Load initial data
    loadSystemMetrics()

    return () => {
      newSocket.close()
    }
  }, [status, router])

  const loadSystemMetrics = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/metrics`)
      
      if (response.ok) {
        const data = await response.json()
        if (data.metrics && data.metrics.length > 0) {
          setSystemMetrics(data.metrics[0])
        }
      }
    } catch (error) {
      console.error('Error loading metrics:', error)
    }
  }

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!uploadedFile) {
      toast.error('Please select a file')
      return
    }

    setIsLoading(true)
    
    try {
      const formData = new FormData()
      formData.append('file', uploadedFile)

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/upload`, {
        method: 'POST',
        body: formData
      })

      const data = await response.json()
      
      if (data.success) {
        toast.success('Data uploaded and processed successfully!')
        setUploadedFile(null)
        // Reset file input
        const fileInput = document.getElementById('data-file') as HTMLInputElement
        if (fileInput) fileInput.value = ''
      } else {
        toast.error(`Error: ${data.error}`)
      }
    } catch (error) {
      toast.error('Upload failed')
    } finally {
      setIsLoading(false)
    }
  }

  const generateForecast = async () => {
    setIsLoading(true)
    
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/forecast`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      const data = await response.json()
      
      if (data.success) {
        setForecastData(data.forecast)
        toast.success('Forecast generated successfully!')
      } else {
        toast.error(`Error: ${data.error}`)
      }
    } catch (error) {
      toast.error('Forecast generation failed')
    } finally {
      setIsLoading(false)
    }
  }

  const refreshData = () => {
    loadSystemMetrics()
    toast.success('Data refreshed')
  }

  if (status === 'loading') {
    return (
      <div className="d-flex justify-content-center align-items-center vh-100">
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
      </div>
    )
  }

  return (
    <>
      <Head>
        <title>Dashboard - DAN_G Refinery Platform</title>
        <meta name="description" content="Refinery forecasting dashboard" />
      </Head>

      {/* Navigation */}
      <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
        <div className="container-fluid">
          <span className="navbar-brand">
            <FaOilCan className="me-2" />DAN_G Refinery Platform
          </span>
          <div className="navbar-nav ms-auto">
            <span className="navbar-text me-3">
              <span className={`badge ${systemMetrics?.status === 'healthy' ? 'bg-success' : 'bg-warning'}`}>
                {systemMetrics?.status || 'Unknown'}
              </span>
              Real-time Monitoring
            </span>
            <Button variant="outline-light" onClick={() => router.push('/api/auth/signout')}>
              <FaSignOutAlt className="me-1" />Logout
            </Button>
          </div>
        </div>
      </nav>

      <Container fluid className="mt-4">
        {/* System Status Row */}
        <Row className="mb-4">
          <Col xs={12}>
            <Card className="dashboard-card">
              <Card.Header className="d-flex justify-content-between align-items-center">
                <h5 className="mb-0"><FaTachometerAlt className="me-2" />System Status</h5>
                <small className="text-muted">Last updated: {new Date().toLocaleTimeString()}</small>
              </Card.Header>
              <Card.Body>
                <Row>
                  <Col md={3}>
                    <div className="metric-card p-3 text-center text-white rounded">
                      <h4>{systemMetrics?.cpu_usage || 0}%</h4>
                      <p className="mb-0">CPU Usage</p>
                    </div>
                  </Col>
                  <Col md={3}>
                    <div className="metric-card p-3 text-center text-white rounded">
                      <h4>{systemMetrics?.memory_usage || 0}%</h4>
                      <p className="mb-0">Memory Usage</p>
                    </div>
                  </Col>
                  <Col md={3}>
                    <div className="metric-card p-3 text-center text-white rounded">
                      <h4>{systemMetrics?.active_forecasts || 0}</h4>
                      <p className="mb-0">Active Forecasts</p>
                    </div>
                  </Col>
                  <Col md={3}>
                    <div className="metric-card p-3 text-center text-white rounded">
                      <h4>
                        <span className={`badge ${systemMetrics?.status === 'healthy' ? 'bg-success' : 'bg-warning'}`}>
                          {systemMetrics?.status || 'unknown'}
                        </span>
                      </h4>
                      <p className="mb-0">System Status</p>
                    </div>
                  </Col>
                </Row>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Main Content Row */}
        <Row>
          {/* Left Column - Controls */}
          <Col lg={4}>
            {/* Data Upload Card */}
            <Card className="dashboard-card mb-4">
              <Card.Header>
                <h5 className="mb-0"><FaUpload className="me-2" />Data Upload</h5>
              </Card.Header>
              <Card.Body>
                <Form onSubmit={handleFileUpload}>
                  <Form.Group className="mb-3">
                    <Form.Label>Select Data File</Form.Label>
                    <Form.Control
                      type="file"
                      id="data-file"
                      accept=".csv,.xlsx,.xls,.json"
                      onChange={(e) => setUploadedFile(e.target.files?.[0] || null)}
                      required
                    />
                    <Form.Text>Supported formats: CSV, Excel, JSON</Form.Text>
                  </Form.Group>
                  <Button type="submit" variant="primary" className="w-100" disabled={isLoading}>
                    <FaUpload className="me-2" />
                    {isLoading ? 'Processing...' : 'Upload & Process'}
                  </Button>
                </Form>
              </Card.Body>
            </Card>

            {/* Quick Actions Card */}
            <Card className="dashboard-card mb-4">
              <Card.Header>
                <h5 className="mb-0"><FaCogs className="me-2" />Quick Actions</h5>
              </Card.Header>
              <Card.Body>
                <div className="d-grid gap-2">
                  <Button variant="success" onClick={generateForecast} disabled={isLoading}>
                    <FaChartLine className="me-2" />Generate Forecast
                  </Button>
                  <Button variant="info" onClick={refreshData}>
                    <FaSyncAlt className="me-2" />Refresh Data
                  </Button>
                </div>
              </Card.Body>
            </Card>
          </Col>

          {/* Right Column - Charts */}
          <Col lg={8}>
            {/* Forecast Results Card */}
            <Card className="dashboard-card mb-4">
              <Card.Header>
                <h5 className="mb-0"><FaChartLine className="me-2" />Forecast Results</h5>
              </Card.Header>
              <Card.Body>
                {forecastData.length > 0 ? (
                  <Plot
                    data={[
                      {
                        x: forecastData.map(d => d.date),
                        y: forecastData.map(d => d.forecast_value),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Forecast',
                        line: { color: '#007bff', width: 3 }
                      },
                      {
                        x: forecastData.map(d => d.date),
                        y: forecastData.map(d => d.confidence_upper),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Upper Bound',
                        line: { color: 'rgba(0,123,255,0.3)', width: 1 },
                        showlegend: false
                      },
                      {
                        x: forecastData.map(d => d.date),
                        y: forecastData.map(d => d.confidence_lower),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Lower Bound',
                        line: { color: 'rgba(0,123,255,0.3)', width: 1 },
                        fill: 'tonexty',
                        fillcolor: 'rgba(0,123,255,0.1)',
                        showlegend: false
                      }
                    ]}
                    layout={{
                      title: '7-Day Yield Forecast',
                      xaxis: { title: 'Date' },
                      yaxis: { title: 'Yield Value' },
                      hovermode: 'x unified',
                      margin: { t: 50, r: 50, b: 50, l: 50 }
                    }}
                    style={{ width: '100%', height: '400px' }}
                  />
                ) : (
                  <div className="text-center text-muted">
                    <FaChartLine size={48} className="mb-3" />
                    <p>Upload data and generate forecast to see results</p>
                  </div>
                )}
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </Container>

      <style jsx>{`
        .dashboard-card {
          transition: transform 0.2s;
          border-left: 4px solid #007bff;
        }
        .dashboard-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .metric-card {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
      `}</style>
    </>
  )
}
