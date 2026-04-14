#!/bin/bash
# Cloudera AI Workbench build script — installs project dependencies.
set -e

pip install --no-cache-dir -r requirements.txt
