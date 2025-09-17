from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
import os
import uuid
import shutil
import traceback
from datetime import datetime

# Import refactored processing functions and S3 utilities
from app.processing.extraction import run_extraction
from app.processing.etl import run_etl_pipeline
from app.core.s3_utils import get_s3_client, download_s3_file, upload_df_to_s3

# --- FastAPI App Initialization ---
app = FastAPI(title="Refinery ETL Pipeline API")

# --- Request Model ---
class ETLTriggerRequest(BaseModel):
    s3_bucket: str
    lab_data_key: str = Field(..., description="S3 key for Lab.csv (e.g., 'raw_data/JUNE,2024/Lab.csv')")
    ts_monitoring_key: str = Field(..., description="S3 key for TS Monitoring Tags.csv")
    blend_data_key: str = Field(..., description="S3 key for blend.csv")
    output_prefix: str = Field("processed_data/", description="S3 prefix for saving the TFT-ready output CSV")

# --- Background Task Function ---
def run_etl_background(request_id: str, s3_bucket: str, lab_key: str, ts_key: str, blend_key: str, output_prefix: str):
    """The actual ETL process run in the background."""
    print(f"[{request_id}] Background ETL task started at {datetime.now()}")
    s3_client = get_s3_client()
    if not s3_client:
        print(f"[{request_id}] Failed to initialize S3 client. Aborting task.")
        # Log failure status somewhere if needed
        return

    temp_dir = None # Initialize to ensure cleanup happens
    try:
        # 1. Create a unique temporary directory for this run
        temp_dir = f"/tmp/{request_id}" # Use /tmp on Heroku/Linux
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[{request_id}] Created temporary directory: {temp_dir}")

        # 2. Download files from S3
        lab_local_path = os.path.join(temp_dir, "Lab.csv")
        ts_local_path = os.path.join(temp_dir, "TS_Monitoring_Tags.csv")
        blend_local_path = os.path.join(temp_dir, "blend.csv")

        dl_success = True
        dl_success &= download_s3_file(s3_client, s3_bucket, lab_key, lab_local_path)
        dl_success &= download_s3_file(s3_client, s3_bucket, ts_key, ts_local_path)
        dl_success &= download_s3_file(s3_client, s3_bucket, blend_key, blend_local_path)

        if not dl_success:
            raise RuntimeError(f"[{request_id}] Failed to download one or more required files from S3.")

        # 3. Run Extraction
        extracted_data = run_extraction(lab_local_path, ts_local_path, blend_local_path)
        if not extracted_data:
            raise ValueError(f"[{request_id}] Data extraction resulted in empty data.")

        # 4. Run ETL Preprocessing
        darts_ready_df = run_etl_pipeline(extracted_data)
        if darts_ready_df is None or darts_ready_df.empty:
             raise ValueError(f"[{request_id}] ETL pipeline did not produce a valid DataFrame.")

        # 5. Upload Result to S3
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"tft_ready_data_{timestamp}.csv"
        output_s3_key = os.path.join(output_prefix, output_filename).replace("\\", "/") # Ensure S3 path uses forward slashes
        upload_success = upload_df_to_s3(s3_client, darts_ready_df, s3_bucket, output_s3_key, index=True)

        if upload_success:
            print(f"[{request_id}] Successfully processed and uploaded result to s3://{s3_bucket}/{output_s3_key}")
            # Log success status somewhere if needed
        else:
            print(f"[{request_id}] Failed to upload processed data to S3.")
            # Log failure status

    except Exception as e:
        print(f"!!! [{request_id}] ETL Background Task FAILED !!!")
        print(f"Error: {e}")
        traceback.print_exc()
        # Log failure status

    finally:
        # 6. Cleanup temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"[{request_id}] Cleaned up temporary directory: {temp_dir}")
            except Exception as ce:
                print(f"[{request_id}] Error cleaning up temp directory {temp_dir}: {ce}")
        print(f"[{request_id}] Background ETL task finished at {datetime.now()}")

# --- API Endpoint ---
@app.post("/trigger-etl", status_code=202) # 202 Accepted: Request taken, processing in background
async def trigger_etl_pipeline(request: ETLTriggerRequest, background_tasks: BackgroundTasks):
    """
    Triggers the ETL pipeline to download data from S3, process it,
    and upload the TFT-ready data back to S3. Runs as a background task.
    """
    request_id = str(uuid.uuid4())
    print(f"Received ETL trigger request {request_id}. Bucket: {request.s3_bucket}")

    # Add the long-running task to the background
    background_tasks.add_task(
        run_etl_background,
        request_id=request_id,
        s3_bucket=request.s3_bucket,
        lab_key=request.lab_data_key,
        ts_key=request.ts_monitoring_key,
        blend_key=request.blend_data_key,
        output_prefix=request.output_prefix
    )

    return {"message": "ETL pipeline triggered successfully.", "request_id": request_id}

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

# --- Running the app (for local testing) ---
# if __name__ == "__main__":
#     import uvicorn
#     # Note: Run this from the *parent directory* of 'app/'
#     # Example: python -m uvicorn app.main:app --reload --port 8000
#     # uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get('PORT', 8000)), reload=True)