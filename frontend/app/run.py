import os

# Set environment variable to handle OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 to localhost
        port=8051,
        reload=True,
        workers=1,  # Explicitly set number of workers
    )
