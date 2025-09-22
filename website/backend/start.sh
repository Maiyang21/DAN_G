#!/bin/bash
echo "Starting DAN_G Refinery Backend..."
cd /app
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py