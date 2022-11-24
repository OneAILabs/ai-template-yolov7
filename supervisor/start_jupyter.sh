#!/bin/bash
set -e
echo $PASSWORD

jupyter-lab \
  --allow-root \
  --no-browser \
  --ip 0.0.0.0 --port 8888 \
  --ServerApp.root_dir="/jupyter-example" \
  --config ${JUPYTER_CONFIG_DIR}/jupyter_lab_config.py
