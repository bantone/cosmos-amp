#!/bin/bash
# Launches the Streamlit app inside Cloudera AI Workbench.
# CDSW_APP_PORT is injected automatically by the platform.

set -e

PORT=${CDSW_APP_PORT:-8501}

exec streamlit run cosmos_app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
