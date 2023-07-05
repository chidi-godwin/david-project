"""Sever side entry point for the application."""
import os

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pydantic import BaseModel
import json

matplotlib.use("Agg")


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the directory for storing uploaded files
UPLOADS_DIRECTORY = "uploads"
INPUT_COLUMNS = [
    "Depth (ft)",
    "Pressure(Psi)",
    "Test flow rate(MCF/D)",
    "Compressibility",
    "Density of gas",
    "Tubing ID(in)",
    "Casing ID(in)",
    "Tubing OD(in)",
    "Flow Area",
    "Density of Liquid",
    "Temperature",
    "Surface Tension",
]


def read_csv(path):
    columns = pd.MultiIndex.from_tuples(
        [
            ("Turner's Model", "Critical Velocity(ft/s)"),
            ("Turner's Model", "Gas flow rate(mcf/days)"),
            ("Coleman's Model", "Critical Velocity(ft/s)"),
            ("Coleman's Model", "Gas flow rate(mcf/days)"),
            ("Li's Model", "Critical Velocity(ft/s)"),
            ("Li's Model", "Gas flow rate(mcf/days)"),
            ("Nosseir's Model", "Critical Velocity(ft/s)"),
            ("Nosseir's Model", "Gas flow rate(mcf/days)"),
            ("Actual Data", "Pressure(Psi)"),
            ("Actual Data", "Gas flow rate(mcf/days)"),
            ("", "Well Status"),
        ]
    )
    df = pd.read_csv(path, header=[0, 1])
    df.columns = columns

    return df


def process_csv(data: list[list[float]], columns):
    original_data = data
    def calculate_model_values(df, model_name):
        model_constant = {
            "turner": 1.92,
            "li": 0.7241,
            "coleman": 1.593,
            "nosseir": 1.938,
        }

        critical_velocity = model_constant[model_name] * (
            (
                (
                    df["Surface Tension"]
                    * abs((df["Density of Liquid"] - df["Density of gas"]))
                )
                / (df["Density of gas"] ** 2)
            )
            ** 0.25
        )
        critical_flow_rate = 3060 * (
            (df["Pressure(Psi)"] * critical_velocity * df["Flow Area"])
            / ((df["Temperature"]) * df["Compressibility"])
        )

        return pd.DataFrame(
            {
                "Critical Velocity(ft/s)": critical_velocity,
                "Gas flow rate(mcf/days)": critical_flow_rate,
            }
        )
    
    df = pd.DataFrame(data, columns=INPUT_COLUMNS)
    
    for column in df.columns:
        if df[column].dtype == "object":
            # Remove commas from string values and convert to float
            df[column] = df[column].str.replace(",", "").astype(float)
        else:
            df[column] = df[column].astype(float)

    multi_column = [
        ("Turner's Model", "Critical Velocity(ft/s)"),
        ("Turner's Model", "Gas flow rate(mcf/days)"),
        ("Coleman's Model", "Critical Velocity(ft/s)"),
        ("Coleman's Model", "Gas flow rate(mcf/days)"),
        ("Li's Model", "Critical Velocity(ft/s)"),
        ("Li's Model", "Gas flow rate(mcf/days)"),
        ("Nosseir's Model", "Critical Velocity(ft/s)"),
        ("Nosseir's Model", "Gas flow rate(mcf/days)"),
        ("Actual Data", "Pressure(Psi)"),
        ("Actual Data", "Gas flow rate(mcf/days)"),
    ]
    data_columns = pd.MultiIndex.from_tuples(multi_column)
    model_data = []

    for model in ["turner", "li", "coleman", "nosseir"]:
        model_values = calculate_model_values(df, model)
        model_data.append(model_values)

    actual_data = df[["Pressure(Psi)", "Test flow rate(MCF/D)"]]
    model_data.append(actual_data)

    data = pd.concat(model_data, axis=1)
    data.columns = data_columns

    well_status = np.where(
        data.loc[:, ("Actual Data", "Gas flow rate(mcf/days)")]
        > data.loc[:, ("Nosseir's Model", "Gas flow rate(mcf/days)")],
        "Liquid Unloading",
        "Liquid Loading",
    )
    well_status = np.reshape(well_status, (-1, 1))  # Reshape to match the index shape

    well_status_df = pd.DataFrame(
        well_status,
        columns=pd.MultiIndex.from_tuples([("", "Well Status")]),
        index=data.index,
    )
    result = pd.concat([data, well_status_df], axis=1)

    # Create the uploads directory if it doesn't exist
    os.makedirs(UPLOADS_DIRECTORY, exist_ok=True)

    result.to_csv(
        os.path.join(UPLOADS_DIRECTORY, "data.csv"), index=False, header=True, sep=","
    )
    df.describe().to_csv(
        os.path.join(UPLOADS_DIRECTORY, "stats.csv"), index=False, header=True, sep=","
    )
    
    return original_data


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

    df = pd.read_csv(file.file)
    df.fillna(0, inplace=True)

    

    return {
        "data": process_csv(df.values.tolist(), INPUT_COLUMNS),
    }

class Data(BaseModel):
    data: list[list[float]]

@app.patch("/submit-file/")
async def update_data(data: Data):
    """
    endpoint to receive the file from the form.
    """
    
    processed_data = process_csv(data.data, INPUT_COLUMNS)

    return {
        "data": processed_data,
    }


@app.post("/analysis/")
async def analysis(request: Request):
    return templates.TemplateResponse("analysis.html", {"request": request})


@app.post("/stats-params/")
async def stats_params(request: Request):
    """
    endpoint to display stats params from sheet 1
    """
    df = read_csv("./uploads/data.csv")
    original = pd.read_csv("./uploads/stats.csv")
    stats = df.describe()
    table1 = original.to_html(index=False, classes="table table-dark table-striped")
    table2 = stats.to_html(index=False, classes="table table-dark table-striped")
    return templates.TemplateResponse(
        "stats_params.html",
        {
            "request": request,
            "table1": table1,
            "table2": table2,
            "title": "Statistical Parameters",
        },
    )


@app.post("/actual-data/")
async def actual_data(request: Request):
    """
    endpoint to display actual data from sheet 3
    """

    df = read_csv("./uploads/data.csv")

    table = df.to_html(index=False, classes="table table-dark table-striped")

    return templates.TemplateResponse(
        "results.html", {"request": request, "table": table, "title": "Results"}
    )


@app.post("/display-chart")
def display_chart(request: Request):
    """
    endpoint to display chart
    """

    # create the crossplot
    result = read_csv("./uploads/data.csv")

    # Extract the required columns for the crossplot
    crossplot_data = result[
        [
            (model, "Gas flow rate(mcf/days)")
            for model in [
                "Turner's Model",
                "Coleman's Model",
                "Li's Model",
                "Nosseir's Model",
                "Actual Data",
            ]
        ]
    ]
    crossplot_data.columns = crossplot_data.columns.droplevel(
        level=1
    )  # Remove the second level of the MultiIndex column

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot each model's gas flow rate against the pressure of the actual data
    for model in ["Turner's Model", "Coleman's Model", "Li's Model", "Nosseir's Model"]:
        plt.scatter(
            result[("Actual Data", "Pressure(Psi)")], crossplot_data[model], label=model
        )

    # Add labels and legend
    plt.xlabel("Pressure (Psi)")
    plt.ylabel("Gas flow rate (mcf/days)")
    plt.legend()

    # save the plot
    plt.savefig("./static/crossplot.png")

    # create the gridplot
    crossplot_data = result[
        [
            ("Actual Data", "Pressure(Psi)"),
            ("Actual Data", "Gas flow rate(mcf/days)"),
            ("", "Well Status"),
        ]
    ].copy()

    # Determine the number of models
    num_models = 4

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Flatten the axes array to make it easier to iterate over
    axes = axes.flatten()

    # Plot each model's gas flow rate against the pressure of the actual data, with different colors for 'Liquid Loading' and 'Liquid Unloading'
    well_status_colors = {"Liquid Loading": "blue", "Liquid Unloading": "red"}
    for i, model in enumerate(
        ["Turner's Model", "Coleman's Model", "Li's Model", "Nosseir's Model"]
    ):
        ax = axes[i]  # Get the current subplot
        model_gas_flow = result[(model, "Gas flow rate(mcf/days)")]
        well_status = crossplot_data[("", "Well Status")]
        color = [well_status_colors.get(status, "black") for status in well_status]
        ax.scatter(
            crossplot_data[("Actual Data", "Pressure(Psi)")],
            model_gas_flow,
            c=color,
            label=model,
        )
        ax.set_xlabel("Pressure (Psi)")
        ax.set_ylabel("Gas flow rate (mcf/days)")
        ax.set_title(model)
        ax.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    color="w",
                    label="Liquid Loading",
                    markerfacecolor="blue",
                    markersize=8,
                ),
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    color="w",
                    label="Liquid Unloading",
                    markerfacecolor="red",
                    markersize=8,
                ),
            ],
            title="Well Status",
            loc="upper right",
        )

    # Remove any unused subplots
    if num_models < len(axes):
        for j in range(num_models, len(axes)):
            fig.delaxes(axes[j])

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Save the plot as a PNG file
    fig.savefig("./static/gridplot.png")

    # Close the figure to free up memory
    plt.close(fig)

    return templates.TemplateResponse("chart.html", {"request": request})


if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    columns = pd.MultiIndex.from_tuples(
        [
            ("Turner's Model", "Critical Velocity(ft/s)"),
            ("Turner's Model", "Gas flow rate(mcf/days)"),
            ("Coleman's Model", "Critical Velocity(ft/s)"),
            ("Coleman's Model", "Gas flow rate(mcf/days)"),
            ("Li's Model", "Critical Velocity(ft/s)"),
            ("Li's Model", "Gas flow rate(mcf/days)"),
            ("Nosseir's Model", "Critical Velocity(ft/s)"),
            ("Nosseir's Model", "Gas flow rate(mcf/days)"),
            ("Actual Data", "Pressure(Psi)"),
            ("Actual Data", "Gas flow rate(mcf/days)"),
            ("", "Well Status"),
        ]
    )
    df = pd.read_csv("./uploads/data.csv", header=[0, 1])
    df.columns = columns
    print(df)
