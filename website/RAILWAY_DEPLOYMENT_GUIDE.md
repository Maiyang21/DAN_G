# Railway Deployment Guide - Multiple Options

## 🚀 Railway Deployment Options

I've created multiple deployment configurations to ensure Railway can properly detect and deploy your backend:

### Option 1: Railway.toml (Recommended)
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "python app.py"
healthcheckPath = "/api/health"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 10

[env]
PYTHON_VERSION = "3.11.0"
```

### Option 2: Nixpacks.toml
```toml
[phases.setup]
nixPkgs = ["python311", "pip"]

[phases.install]
cmds = [
    "pip install --upgrade pip",
    "pip install -r requirements.txt"
]

[phases.build]
cmds = ["echo 'Build phase complete'"]

[start]
cmd = "python app.py"
```

### Option 3: Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

### Option 4: Procfile + start.sh
```
web: python app.py
```

## 📋 Deployment Steps

### Method 1: Using Railway Dashboard
1. Go to [Railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `DAN_G` repository
4. **Important**: Set the root directory to `website/backend`
5. Railway will automatically detect the configuration files
6. Set environment variables (see below)

### Method 2: Using Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Link to existing project
railway link

# Deploy
railway up
```

## 🔧 Environment Variables

Set these in Railway dashboard:

```bash
# Database
DATABASE_URL=postgresql://username:password@host:port/database

# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name
SAGEMAKER_ENDPOINT=your_sagemaker_endpoint

# Flask Configuration
SECRET_KEY=your_secret_key_here
FLASK_ENV=production

# CORS
CORS_ORIGINS=https://your-frontend.vercel.app
```

## 🔍 Troubleshooting

### If Railway still can't detect the app:

1. **Check Root Directory**: Make sure Railway is looking at `website/backend` not the root
2. **Verify Files**: Ensure all config files are in the backend directory
3. **Check Logs**: Look at Railway build logs for specific errors
4. **Try Dockerfile**: Railway should detect the Dockerfile and use it

### Common Issues:

**Issue**: "Script start.sh not found"
**Solution**: Railway should now detect `railway.toml` or `nixpacks.toml` instead

**Issue**: "Cannot determine build method"
**Solution**: The `package.json` file helps Railway understand this is a Python project

**Issue**: "Port binding error"
**Solution**: Make sure your Flask app binds to `0.0.0.0:5000` (which it does)

## ✅ Verification

After deployment, test these endpoints:
- `GET /api/health` - Health check
- `GET /api/forecast` - Forecast endpoint
- `POST /api/upload` - File upload

## 📁 File Structure

```
website/backend/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── start.sh              # Startup script
├── Procfile              # Process definition
├── runtime.txt           # Python version
├── railway.toml          # Railway configuration
├── nixpacks.toml         # Nixpacks configuration
├── Dockerfile            # Docker configuration
├── package.json          # Project metadata
└── test_deployment.py    # Test script
```

## 🎯 Next Steps

1. Deploy to Railway using any of the methods above
2. Get the Railway URL (e.g., `https://your-app.railway.app`)
3. Update your frontend environment variables with the Railway URL
4. Deploy frontend to Vercel
5. Test the full application

The multiple configuration files ensure Railway will definitely be able to deploy your backend! 🚀
