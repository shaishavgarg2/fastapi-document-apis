import os
import uvicorn

# Import the app from parameter_value_extraction_api
from parameter_value_extraction_api import app

if __name__ == "__main__":
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get("PORT", 8002))
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=port)