import subprocess
import sys

subprocess.run([
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "streamlit_app.py",
    "--server.port=7860",
    "--server.address=0.0.0.0",
])