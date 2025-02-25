from fastapi import FastAPI
from app.routes.analyze import router

app = FastAPI(
    title="Deformation Analysis API",
    description="API for analyzing image deformation using a trained model",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8082, reload=True)
