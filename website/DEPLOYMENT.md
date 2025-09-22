# DAN_G Platform Deployment Guide

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vercel        │    │   Railway       │    │   AWS           │
│   (Frontend)    │    │   (Backend)     │    │   (ML & Data)   │
│   - Next.js     │───▶│   - Flask API   │───▶│   - SageMaker   │
│   - Static      │    │   - PostgreSQL  │    │   - S3 Storage  │
│   - Auth UI     │    │   - Redis       │    │   - RDS         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

1. **AWS Account** with SageMaker and S3 access
2. **Vercel Account** for frontend hosting
3. **Railway Account** for backend hosting
4. **GitHub Repository** for code deployment

## Step 1: AWS Setup

### 1.1 Create S3 Bucket
```bash
aws s3 mb s3://dan-g-refinery-data
aws s3api put-bucket-versioning --bucket dan-g-refinery-data --versioning-configuration Status=Enabled
```

### 1.2 Deploy SageMaker Model
```bash
# Create SageMaker endpoint
aws sagemaker create-endpoint-config \
  --endpoint-config-name dan-g-forecasting-config \
  --production-variants VariantName=primary,ModelName=your-model-name,InitialInstanceCount=1,InstanceType=ml.m5.large

aws sagemaker create-endpoint \
  --endpoint-name dan-g-forecasting-endpoint \
  --endpoint-config-name dan-g-forecasting-config
```

### 1.3 Create IAM User
```bash
# Create IAM user with policies
aws iam create-user --user-name dan-g-platform
aws iam attach-user-policy --user-name dan-g-platform --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam attach-user-policy --user-name dan-g-platform --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

## Step 2: Railway Backend Deployment

### 2.1 Connect Repository
1. Go to [Railway](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Choose the `backend` folder

### 2.2 Configure Environment Variables
```bash
# Database
DATABASE_URL=postgresql://username:password@host:port/database

# Redis
REDIS_URL=redis://username:password@host:port

# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
AWS_S3_BUCKET=dan-g-refinery-data
AWS_SAGEMAKER_ENDPOINT=dan-g-forecasting-endpoint

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key

# CORS
CORS_ORIGINS=https://your-vercel-app.vercel.app
```

### 2.3 Deploy
Railway will automatically deploy when you push to the main branch.

## Step 3: Vercel Frontend Deployment

### 3.1 Connect Repository
1. Go to [Vercel](https://vercel.com)
2. Click "New Project" → "Import Git Repository"
3. Select your repository
4. Set Root Directory to `frontend`

### 3.2 Configure Environment Variables
```bash
# API Configuration
NEXT_PUBLIC_API_URL=https://your-railway-app.railway.app

# Authentication
NEXTAUTH_URL=https://your-vercel-app.vercel.app
NEXTAUTH_SECRET=your-nextauth-secret
```

### 3.3 Deploy
Vercel will automatically deploy when you push to the main branch.

## Step 4: Domain Configuration

### 4.1 Custom Domain (Optional)
1. **Vercel**: Add custom domain in project settings
2. **Railway**: Add custom domain in project settings
3. Update CORS_ORIGINS in Railway with your custom domain

### 4.2 SSL Certificates
Both Vercel and Railway provide automatic SSL certificates.

## Step 5: Monitoring and Maintenance

### 5.1 Health Checks
- **Frontend**: `https://your-vercel-app.vercel.app`
- **Backend**: `https://your-railway-app.railway.app/api/health`

### 5.2 Logs
- **Vercel**: View logs in Vercel dashboard
- **Railway**: View logs in Railway dashboard
- **AWS**: CloudWatch logs for SageMaker

### 5.3 Monitoring
- Set up alerts for system health
- Monitor API response times
- Track error rates and performance

## Environment Variables Reference

### Backend (Railway)
```bash
# Database
DATABASE_URL=postgresql://...

# Redis
REDIS_URL=redis://...

# AWS
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_S3_BUCKET=dan-g-refinery-data
AWS_SAGEMAKER_ENDPOINT=dan-g-forecasting-endpoint

# Security
SECRET_KEY=...
JWT_SECRET_KEY=...

# CORS
CORS_ORIGINS=https://your-vercel-app.vercel.app

# Optional
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=...
MAIL_PASSWORD=...
```

### Frontend (Vercel)
```bash
# API
NEXT_PUBLIC_API_URL=https://your-railway-app.railway.app

# Auth
NEXTAUTH_URL=https://your-vercel-app.vercel.app
NEXTAUTH_SECRET=...
```

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Check CORS_ORIGINS in Railway
   - Ensure frontend URL is included

2. **Database Connection Issues**
   - Verify DATABASE_URL in Railway
   - Check PostgreSQL service status

3. **AWS Connection Issues**
   - Verify AWS credentials
   - Check IAM permissions
   - Ensure SageMaker endpoint is active

4. **Authentication Issues**
   - Check JWT_SECRET_KEY
   - Verify NEXTAUTH_SECRET
   - Ensure API URLs are correct

### Debug Commands

```bash
# Check Railway logs
railway logs

# Check Vercel logs
vercel logs

# Test API endpoints
curl https://your-railway-app.railway.app/api/health

# Test AWS connection
aws s3 ls s3://dan-g-refinery-data
```

## Cost Optimization

### Vercel
- Free tier: 100GB bandwidth/month
- Pro: $20/month for additional resources

### Railway
- Starter: $5/month for basic resources
- Pro: $20/month for production resources

### AWS
- SageMaker: Pay per inference
- S3: Pay per storage and requests
- RDS: Pay per instance hours

## Security Best Practices

1. **Environment Variables**
   - Never commit secrets to git
   - Use strong, unique passwords
   - Rotate keys regularly

2. **API Security**
   - Use HTTPS everywhere
   - Implement rate limiting
   - Validate all inputs

3. **Database Security**
   - Use connection pooling
   - Enable SSL connections
   - Regular backups

4. **AWS Security**
   - Use IAM roles with minimal permissions
   - Enable CloudTrail logging
   - Use VPC for SageMaker if needed

## Scaling Considerations

### Horizontal Scaling
- Railway: Auto-scaling based on traffic
- Vercel: Edge functions for global distribution
- AWS: SageMaker auto-scaling

### Vertical Scaling
- Upgrade Railway plan for more resources
- Use larger SageMaker instances
- Optimize database queries

## Backup and Recovery

### Database Backups
- Railway: Automatic daily backups
- Manual: Export data regularly

### Code Backups
- GitHub: Primary code repository
- Vercel: Automatic deployments
- Railway: Automatic deployments

### Data Backups
- S3: Versioning enabled
- Cross-region replication for critical data

## Support and Maintenance

### Monitoring
- Set up uptime monitoring
- Track performance metrics
- Monitor error rates

### Updates
- Regular dependency updates
- Security patches
- Feature updates

### Documentation
- Keep deployment docs updated
- Document any custom configurations
- Maintain runbooks for common issues


