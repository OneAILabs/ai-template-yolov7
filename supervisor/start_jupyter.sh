#!/bin/bash
set -e
echo $PASSWORD
jupyter notebook --notebook-dir=/yolov7 --ip='*' --port=8888 --allow-root --no-browser --NotebookApp.password="$(echo $PASSWORD | python -c 'from notebook.auth import passwd;print(passwd(input()))')"
