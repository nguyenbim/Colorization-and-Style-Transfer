import uvicorn
from fastapi import Depends, FastAPI


from API import router

app = FastAPI()
app.include_router(router)

@app.get("/")
def root_page():
    return {"message": "Welcome to Test API"}

if __name__ == "__main__":
    uvicorn.run(app)    