# server/app.py
import uvicorn
from app.api_server import app

def main():
    """Entry point for OpenEnv multi‑mode deployment."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()