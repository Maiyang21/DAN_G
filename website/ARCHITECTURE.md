# DAN_G Production Architecture

## System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vercel        │    │   Railway       │    │   AWS           │
│   (Frontend)    │    │   (Backend)     │    │   (ML & Data)   │
│   - React UI    │───▶│   - Flask API   │───▶│   - SageMaker   │
│   - Static      │    │   - ETL Pipeline│    │   - S3 Storage  │
│   - Auth UI     │    │   - Database    │    │   - RDS         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Component Details

### Frontend (Vercel)
- **React/Next.js** application
- **Static hosting** with CDN
- **Serverless functions** for API calls
- **Authentication UI** with Auth0/NextAuth
- **Real-time updates** via WebSocket

### Backend (Railway)
- **Flask API** server
- **PostgreSQL** database
- **Redis** for caching and sessions
- **ETL Pipeline** for data processing
- **S3 Integration** for data storage
- **WebSocket** for real-time communication

### ML & Data (AWS)
- **SageMaker** for model training and inference
- **S3** for data storage and model artifacts
- **RDS** for production database (optional)
- **API Gateway** for model endpoints

## Data Flow

1. **User uploads data** → Vercel (Frontend)
2. **Data sent to Railway** → Flask API
3. **ETL processing** → Railway (Backend)
4. **Processed data** → S3 (AWS)
5. **Model inference** → SageMaker (AWS)
6. **Results** → Railway → Vercel → User

## Authentication Flow

1. **User login** → Auth0/NextAuth
2. **JWT token** → Railway API
3. **Token validation** → Railway (Backend)
4. **Authorized requests** → AWS services


