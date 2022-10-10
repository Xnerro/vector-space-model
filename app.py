from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import uvicorn
from utils.index import search_data

app = FastAPI(title="Vector Space Model", version="1.0.0")
app.mount("/public", StaticFiles(directory="public"), name="public")
templates = Jinja2Templates(directory="public", autoescape=True)


@app.get("/")
def main_root(request: Request):
    data = pd.read_csv("data/sampah_bandung.csv")
    new_data = {}
    for i, val in enumerate(data["text"].loc[:]):
        new_data[f"doc{i}"] = val
    return templates.TemplateResponse("index.html", {"request": request, "data": new_data})


@app.get("/search")
def search(request: Request, query: str):
    data = pd.read_csv("data/sampah_bandung.csv")
    result = search_data(query, data)
    return templates.TemplateResponse(
        "pages/search.html",
        {"request": request, "data": data["text"].to_dict(), "query": query, "result": result.argsort()[-10:][::-1]},
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
