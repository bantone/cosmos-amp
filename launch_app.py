import subprocess
import os

port = os.environ.get('CDSW_APP_PORT', '8080')
subprocess.run([
    'streamlit', 'run', 'app.py',
    '--server.port', port,
    '--server.address', '127.0.0.1'
])
