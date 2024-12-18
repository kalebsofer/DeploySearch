import os

# Handle OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8051,
        reload=True,
        workers=1,
    )
