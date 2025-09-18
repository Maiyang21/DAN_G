# DAN_G Platform - Simplified Setup Guide

## ✅ **Authentication Simplified**

The platform now uses **simple session-based authentication** instead of complex JWT tokens.

### **What Changed:**

1. **Removed JWT complexity** - No more token management
2. **Simple session authentication** - Uses Flask sessions
3. **No role-based access** - All users have same permissions
4. **Direct API calls** - Frontend calls backend directly

## 🚀 **Quick Setup**

### **1. Frontend (Vercel)**
```bash
# Environment Variables in Vercel:
NEXT_PUBLIC_API_URL=https://your-railway-app.railway.app
NEXTAUTH_URL=https://your-vercel-app.vercel.app
NEXTAUTH_SECRET=your-secret-key-here
```

### **2. Backend (Railway)**
```bash
# Environment Variables in Railway:
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_S3_BUCKET=dan-g-refinery-data
AWS_SAGEMAKER_ENDPOINT=dan-g-forecasting-endpoint
CORS_ORIGINS=https://your-vercel-app.vercel.app
```

## 🔐 **Authentication Flow**

### **Simple Login Process:**
1. User enters username/password
2. Frontend calls `/api/auth/login`
3. Backend validates credentials
4. Backend creates session
5. User is logged in

### **Demo Credentials:**
- **Username:** `admin`
- **Password:** `admin123`

## 📁 **Updated File Structure**

```
website/
├── frontend/                 # Next.js (Vercel)
│   ├── pages/
│   │   ├── index.tsx        # Landing page with login
│   │   └── dashboard.tsx    # Main dashboard
│   ├── types/
│   │   └── next-auth.d.ts   # TypeScript definitions
│   └── package.json         # Dependencies
├── backend/                  # Flask (Railway)
│   ├── app/
│   │   ├── api/
│   │   │   ├── simple_auth.py    # Simple authentication
│   │   │   ├── aws_client.py     # AWS integration
│   │   │   ├── etl_processor.py  # Data processing
│   │   │   └── forecasting_engine.py # ML models
│   │   └── app.py           # Main Flask app
│   └── requirements.txt     # Python dependencies
└── vercel.json              # Vercel configuration
```

## 🛠️ **Key Features**

### **Frontend:**
- ✅ **Next.js 13** with TypeScript
- ✅ **Simple authentication** without JWT
- ✅ **Direct API calls** to backend
- ✅ **Real-time updates** via WebSocket
- ✅ **Interactive charts** with Plotly

### **Backend:**
- ✅ **Flask API** with session authentication
- ✅ **AWS SageMaker** integration
- ✅ **S3 data storage**
- ✅ **ETL processing** pipeline
- ✅ **WebSocket** for real-time updates

## 🚀 **Deployment Steps**

### **1. Deploy Backend (Railway)**
```bash
# Connect GitHub repo to Railway
# Set environment variables
# Deploy automatically
```

### **2. Deploy Frontend (Vercel)**
```bash
# Connect GitHub repo to Vercel
# Set root directory to 'frontend'
# Set environment variables
# Deploy automatically
```

### **3. Configure AWS**
```bash
# Create S3 bucket
# Deploy SageMaker model
# Set up IAM permissions
```

## 💡 **Benefits of Simplified Auth**

1. **Easier to understand** - No complex token management
2. **Faster development** - Less authentication code
3. **Simpler debugging** - Clear session-based flow
4. **Better for demos** - Works out of the box
5. **Production ready** - Can be enhanced later

## 🔧 **Development Commands**

### **Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### **Backend:**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

## 📊 **API Endpoints**

### **Authentication:**
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/me` - Get current user

### **Data Processing:**
- `POST /api/upload` - Upload data
- `POST /api/forecast` - Generate forecast
- `GET /api/metrics` - System metrics

### **Health Check:**
- `GET /api/health` - System health

## 🎯 **Ready for Production**

The simplified authentication system is:
- ✅ **Production ready**
- ✅ **Secure** with session management
- ✅ **Scalable** with Railway and Vercel
- ✅ **Maintainable** with clean code
- ✅ **Extensible** for future enhancements

No more JWT complexity - just simple, effective authentication! 🎉
