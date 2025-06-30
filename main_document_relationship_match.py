import os
import uvicorn

# Import the app from document_relationship_match
from document_relationship_match import app

if __name__ == "__main__":
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get("PORT", 8001))
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=port)