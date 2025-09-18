import { useState, useEffect } from 'react'
import { useSession, signIn, signOut } from 'next-auth/react'
import { useRouter } from 'next/router'
import Head from 'next/head'
import Link from 'next/link'
import { Container, Row, Col, Card, Button, Form, Alert } from 'react-bootstrap'
import { FaOilCan, FaChartLine, FaCogs, FaShieldAlt, FaMobileAlt, FaTachometerAlt } from 'react-icons/fa'
import toast from 'react-hot-toast'

export default function Home() {
  const { data: session, status } = useSession()
  const router = useRouter()
  const [loginData, setLoginData] = useState({ username: '', password: '' })
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (session) {
      router.push('/dashboard')
    }
  }, [session, router])

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      const result = await signIn('credentials', {
        username: loginData.username,
        password: loginData.password,
        redirect: false,
      })

      if (result?.error) {
        toast.error('Invalid credentials')
      } else {
        toast.success('Login successful!')
        router.push('/dashboard')
      }
    } catch (error) {
      toast.error('Login failed')
    } finally {
      setIsLoading(false)
    }
  }

  if (status === 'loading') {
    return (
      <div className="d-flex justify-content-center align-items-center vh-100">
        <div className="spinner-border text-primary" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
      </div>
    )
  }

  return (
    <>
      <Head>
        <title>DAN_G Refinery Forecasting Platform</title>
        <meta name="description" content="Advanced AI-powered forecasting and optimization for refinery operations" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      {/* Navigation */}
      <nav className="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
        <div className="container">
          <Link className="navbar-brand" href="/">
            <FaOilCan className="me-2" />DAN_G Refinery Platform
          </Link>
          <div className="navbar-nav ms-auto">
            <Link className="nav-link" href="#features">Features</Link>
            <Link className="nav-link" href="#about">About</Link>
            {session ? (
              <Button variant="outline-light" onClick={() => signOut()}>
                Logout
              </Button>
            ) : (
              <Button variant="primary" onClick={() => document.getElementById('loginModal')?.click()}>
                Login
              </Button>
            )}
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="container">
          <Row className="align-items-center">
            <Col lg={6}>
              <h1 className="display-4 fw-bold mb-4">DAN_G Refinery Forecasting Platform</h1>
              <p className="lead mb-4">
                Advanced AI-powered forecasting and optimization for refinery operations. 
                Predict yields, optimize parameters, and maximize efficiency with real-time insights.
              </p>
              <div className="d-flex gap-3">
                <Button 
                  variant="light" 
                  size="lg" 
                  onClick={() => document.getElementById('loginModal')?.click()}
                >
                  <FaChartLine className="me-2" />Get Started
                </Button>
                <Button 
                  variant="outline-light" 
                  size="lg" 
                  onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
                >
                  Learn More
                </Button>
              </div>
            </Col>
            <Col lg={6}>
              <div className="text-center">
                <FaChartLine size={120} className="opacity-75" />
              </div>
            </Col>
          </Row>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-5">
        <div className="container">
          <Row>
            <Col lg={12} className="text-center mb-5">
              <h2 className="display-5 fw-bold">Platform Features</h2>
              <p className="lead text-muted">Comprehensive tools for refinery optimization and forecasting</p>
            </Col>
          </Row>
          <Row>
            {[
              {
                icon: <FaChartLine size={48} className="text-primary" />,
                title: "AI-Powered Forecasting",
                description: "Advanced machine learning models including XGBoost, Ridge Regression, and ensemble methods for accurate yield predictions."
              },
              {
                icon: <FaCogs size={48} className="text-success" />,
                title: "Real-time Optimization",
                description: "Continuous monitoring and optimization of refinery parameters to maximize yields and operational efficiency."
              },
              {
                icon: <FaTachometerAlt size={48} className="text-info" />,
                title: "Real-time Monitoring",
                description: "Live system monitoring with alerts, performance metrics, and health indicators for optimal platform operation."
              },
              {
                icon: <FaShieldAlt size={48} className="text-danger" />,
                title: "Enterprise Security",
                description: "Robust security features including authentication, data encryption, and secure API endpoints for enterprise deployment."
              },
              {
                icon: <FaMobileAlt size={48} className="text-secondary" />,
                title: "Responsive Design",
                description: "Modern, responsive interface that works seamlessly across desktop, tablet, and mobile devices."
              }
            ].map((feature, index) => (
              <Col lg={4} md={6} key={index} className="mb-4">
                <Card className="h-100 feature-card">
                  <Card.Body className="text-center p-4">
                    <div className="mb-3">{feature.icon}</div>
                    <h5 className="card-title">{feature.title}</h5>
                    <p className="card-text">{feature.description}</p>
                  </Card.Body>
                </Card>
              </Col>
            ))}
          </Row>
        </div>
      </section>

      {/* Login Modal */}
      <div className="modal fade" id="loginModal" tabIndex={-1}>
        <div className="modal-dialog modal-dialog-centered">
          <div className="modal-content">
            <div className="modal-header">
              <h5 className="modal-title">
                <FaChartLine className="me-2" />Login to DAN_G Platform
              </h5>
              <button type="button" className="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div className="modal-body">
              <Form onSubmit={handleLogin}>
                <Form.Group className="mb-3">
                  <Form.Label>Username</Form.Label>
                  <Form.Control
                    type="text"
                    value={loginData.username}
                    onChange={(e) => setLoginData({ ...loginData, username: e.target.value })}
                    required
                  />
                </Form.Group>
                <Form.Group className="mb-3">
                  <Form.Label>Password</Form.Label>
                  <Form.Control
                    type="password"
                    value={loginData.password}
                    onChange={(e) => setLoginData({ ...loginData, password: e.target.value })}
                    required
                  />
                </Form.Group>
                <Button 
                  type="submit" 
                  variant="primary" 
                  className="w-100" 
                  disabled={isLoading}
                >
                  {isLoading ? 'Logging in...' : 'Login'}
                </Button>
              </Form>
              <div className="text-center mt-3">
                <small className="text-muted">
                  Demo credentials: admin / admin123
                </small>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .hero-section {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 100px 0;
          margin-top: 76px;
        }
        .feature-card {
          transition: transform 0.3s;
          border: none;
          box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .feature-card:hover {
          transform: translateY(-5px);
        }
      `}</style>
    </>
  )
}
