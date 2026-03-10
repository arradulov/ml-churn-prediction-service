from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import pandas as pd
import joblib
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Churn Prediction WebApp")

# Статика и шаблоны
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Загружаем модель
model = joblib.load("churn_model.pkl")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Проверяем расширение
    filename = file.filename
    if filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    elif filename.endswith(".txt"):
        # пробуем табуляцию как разделитель
        df = pd.read_csv(file.file, sep="\t")
    else:
        return {"error": "Только CSV или TXT файлы разрешены"}

    # Предсказания
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    df["churn_prediction"] = predictions
    df["churn_probability"] = probabilities

    # Сохраняем результат
    result_file = "predictions.csv"
    df.to_csv(result_file, index=False)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": df.head().to_html(classes="table table-striped"),
        "download_link": f"/download/{result_file}"
    })

@app.get("/download/{filename}")
def download_file(filename: str):
    # Отдаем CSV для скачивания
    return FileResponse(filename, media_type='text/csv', filename=filename)