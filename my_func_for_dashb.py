# Import
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn import  metrics
import numpy as np


def calculate_metrics(y_real, y_pred):
    MAE = round(metrics.mean_absolute_error(y_real, y_pred), 2)
    MBE = round(np.mean(y_real - y_pred), 3)
    MSE = round(metrics.mean_squared_error(y_real, y_pred), 2)
    RMSE = round(np.sqrt(metrics.mean_squared_error(y_real, y_pred)), 2)
    cvRMSE = round(RMSE / np.mean(y_real), 4)
    NMBE = round(MBE / np.mean(y_real), 5)
    return MAE, MBE, MSE, RMSE, cvRMSE, NMBE