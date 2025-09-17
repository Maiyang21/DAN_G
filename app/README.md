# Refinery Forecast App — Autoformer + EBM (AWS)

A comprehensive web application for processing refinery data and generating 7-day forecasts using Autoformer + EBM models deployed on AWS SageMaker.

## 🏗️ Architecture

```
app/
├── main.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── core/
│   └── aws_utils.py      # AWS S3 and SageMaker utilities
├── etl/
│   └── etl_pipeline.py   # ETL pipeline integration
└── README.md             # This file
```

## 🚀 Features

- **Excel File Processing**: Handles complex refinery data from multiple sheets
- **ETL Pipeline Integration**: Uses the existing `DANGlocal_etl_run.py` pipeline
- **AWS Integration**: S3 storage and SageMaker endpoint integration
- **Interactive Visualizations**: Plotly charts for data exploration and forecasts
- **Real-time Processing**: Streamlit-based responsive UI
- **Feature Engineering**: Automatic static feature generation
- **Model Explanations**: EBM feature importance visualization

## 📋 Prerequisites

1. **Python 3.8+**
2. **AWS Account** with appropriate permissions
3. **SageMaker Endpoint** hosting Autoformer + EBM models
4. **S3 Bucket or Access Point** for data storage

## ⚙️ Installation

1. **Clone or navigate to the app directory:**
   ```bash
   cd app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Required AWS configuration
   export AWS_REGION=us-east-1
   export S3_BUCKET=your-bucket-name
   # OR use S3 Access Point (recommended)
   export S3_ACCESS_POINT_ARN=arn:aws:s3:us-east-1:123456789012:accesspoint/your-ap
   export S3_PREFIX=refinery-forecast/input/
   export SM_ENDPOINT=autoformer-ebm-endpoint
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region | `us-east-1` |
| `S3_BUCKET` | S3 bucket name | - |
| `S3_ACCESS_POINT_ARN` | S3 Access Point ARN | - |
| `S3_PREFIX` | S3 key prefix | `refinery-forecast/input/` |
| `SM_ENDPOINT` | SageMaker endpoint name | `autoformer-ebm-endpoint` |

### IAM Permissions

Your AWS credentials need the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket/refinery-forecast/*",
                "arn:aws:s3:us-east-1:123456789012:accesspoint/your-ap/refinery-forecast/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": [
                "arn:aws:sagemaker:us-east-1:123456789012:endpoint/autoformer-ebm-endpoint"
            ]
        }
    ]
}
```

## 🏃‍♂️ Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run main.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Upload your Excel files** containing:
   - **TS Monitoring Tags** sheet
   - **Blends** sheet
   - **Lab** sheet

4. **Run the ETL pipeline** to process your data

5. **Generate forecasts** using the SageMaker endpoint

## 📊 Data Format

### Input Files

The application expects Excel files with the following sheets:

#### TS Monitoring Tags Sheet
- Contains time series data with tags and values
- Supports both formats:
  - **June 2024**: `[S/N, Tag, Date1, Date2, ...]`
  - **February 2025**: `[S/N, DCS Tag, Tag, Date1, Date2, ...]`

#### Blends Sheet
- Contains blend composition data
- Row 0: API values
- Row 1: Sulphur values
- Row 2: Crude Oil%
- Row 3+: Crude type compositions

#### Lab Sheet
- Contains laboratory analysis data

### Output Data

The ETL pipeline generates three main datasets:

1. **Targets**: Product streams and key measurements
2. **Statics**: Blend information and static features
3. **Futures**: Input features for the model

## 🔮 Model Integration

### SageMaker Endpoint Contract

The app sends a JSON payload to your SageMaker endpoint:

```json
{
  "horizon": 7,
  "schema": {
    "time": "date",
    "targets": ["target_RCO_flow", "target_RCO_Yield", ...]
  },
  "data_sample": [
    {"date": "2025-06-01", "target_RCO_flow": 700, ...},
    ...
  ],
  "static": {
    "static_blend_diversity_index": 1.2,
    "static_blend_AGB_mean": 0.2,
    ...
  }
}
```

### Expected Response Format

Your SageMaker endpoint should return:

```json
{
  "forecast": [
    {
      "date": "2025-08-30",
      "target": "target_RCO_flow",
      "p50": 680.2,
      "p10": 650.0,
      "p90": 710.5
    },
    ...
  ],
  "explanations": {
    "static_blend_diversity_index": 0.34,
    "static_blend_AGB_mean": 0.22,
    ...
  }
}
```

## 🛠️ Development

### Project Structure

- `main.py`: Main Streamlit application with UI components
- `config.py`: Configuration management
- `core/aws_utils.py`: AWS service integrations
- `etl/etl_pipeline.py`: ETL pipeline wrapper and utilities

### Adding New Features

1. **New ETL Steps**: Modify `etl/etl_pipeline.py`
2. **UI Components**: Add to `main.py`
3. **AWS Services**: Extend `core/aws_utils.py`
4. **Configuration**: Update `config.py`

### Testing

```bash
# Test ETL pipeline
python -c "from etl.etl_pipeline import process_uploaded_files; print('ETL module loaded successfully')"

# Test AWS utilities
python -c "from core.aws_utils import check_aws_credentials; print('AWS credentials:', check_aws_credentials())"
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Error for DANGlocal_etl_run**:
   - Ensure the `DORC_PROCESS_OPTIMIZER` directory is in the correct location
   - Check Python path configuration

2. **AWS Credentials Not Found**:
   - Verify AWS credentials are configured
   - Check IAM permissions

3. **SageMaker Endpoint Not Responding**:
   - Verify endpoint name and region
   - Check endpoint status in AWS Console

4. **File Upload Issues**:
   - Check file size limits
   - Verify file format (Excel files required)

### Debug Mode

Enable verbose logging by setting:
```bash
export STREAMLIT_LOGGER_LEVEL=debug
```

## 📈 Performance Considerations

- **File Size**: Large Excel files may take time to process
- **Memory Usage**: Monitor memory consumption with large datasets
- **S3 Upload**: Network speed affects upload performance
- **SageMaker**: Endpoint cold starts may cause initial delays

## 🔒 Security

- **AWS Credentials**: Use IAM roles when possible
- **S3 Access**: Implement least-privilege access
- **Data Privacy**: Ensure sensitive refinery data is properly protected
- **Network**: Use VPC endpoints for secure AWS communication

## 📝 License

This project is part of the refinery optimization system. Please ensure compliance with your organization's data handling policies.

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review AWS CloudWatch logs
3. Verify configuration settings
4. Contact your system administrator

---

**Built with ❤️ for refinery optimization**

