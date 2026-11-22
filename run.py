"""
DocuMind Application Runner.
Convenience entrypoint ensuring module discovery and development reloading.
"""

import sys
from pathlib import Path
import uvicorn

# Ensure the root project directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    print(f"Starting DocuMind server from {ROOT_DIR}...")
    uvicorn.run(
        "documind.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
