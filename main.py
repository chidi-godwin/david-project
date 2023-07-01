"""Sever side entry point for the application."""
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index(request: Request):
    """
    Render the index.html template with the request object.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit-file/")
async def submit_file(request: Request, file: UploadFile = File(...)):
    """
    endpoint to receive the file from the form.
    """
    return templates.TemplateResponse('analysis.html', {"request": request})


@app.post('/stats-params/')
async def stats_params(request: Request):
    """
    endpoint to display stats params from sheet 1
    """
    df = pd.read_excel('./static/sheets.xlsx', sheet_name='Sheet2')
    start_cell = (1,2)
    end_cell = (13,13)
    data = df.iloc[start_cell[0]:end_cell[0] + 1, start_cell[1]:end_cell[1] + 1]
    data.columns = data.iloc[0].values
    data = data[1:]
    data.reset_index(drop=True, inplace=True)
    table = data.to_html(index=False, classes='table table-dark table-striped')
    return templates.TemplateResponse("stats_params.html", {"request": request, "table": table, "title": "Statistical Parameters"})


@app.post('/actual-data/')
async def actual_data(request: Request):
    """
    endpoint to display actual data from sheet 3
    """

    df = pd.read_excel('./static/sheets.xlsx', sheet_name='Sheet3')
    start_cell = (0,0)
    end_cell = (108,13)

    data = df.iloc[start_cell[0]:end_cell[0] + 1, start_cell[1]:end_cell[1] + 1]
    multi_column = [('Li\'s Model', 'Critical Velocity(ft/s)'),('Li\'s Model', 'Gas flow rate(mcf/days)'), ('Li\'s Model', 'Model validation'),('Turner\'s Model', 'Critical Velocity(ft/s)'),('Turner\'s Model', 'Gas flow rate(mcf/days)'), ('Turner\'s Model', 'Model validation'), ('Ikepka\'s Model', 'Critical Velocity(ft/s)'),('Ikepka\'s Model', 'Gas flow rate(mcf/days)'), ('Ikepka\'s Model', 'Model validation'), ('Actual Data', 'Wellhead Pressure(Psi)'), ('Actual Data','Gas flow rate(mcf/days)')]
    data_columns = pd.MultiIndex.from_tuples(multi_column)
    data.columns = data_columns
    data = data.iloc[1:, :]
    data.reset_index(drop=True, inplace=True)

    table = data.to_html(index=False, classes='table table-dark table-striped')

    return templates.TemplateResponse("stats_params.html", {"request": request, "table": table, "title": "Actual Data"})

@app.post('/display-chart')
def display_chart(request: Request):
    """
    endpoint to display chart
    """
    return templates.TemplateResponse("chart.html", {"request": request, "image": 'chart.png'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
