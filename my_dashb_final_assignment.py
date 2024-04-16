# Import
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from my_func_for_dashb import calculate_metrics
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import dash_table

# Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Load data
df_23 = pd.read_csv('all_data_2023.csv')
df_20_23 = pd.read_csv('all_data_2020_2023.csv')

# Forecast models
forecast_models = {
    'Linear': 'Linear Regression',
    'Neural': 'Neural Networks',
    'Bootstrapping': 'Bootstrapping',
    'Random Forest': 'Random Forest',
    'Decision Tree': 'Decision Tree',
    'GB': 'Gradient Boosting',
    'XGB': 'Extreme Gradient Boosting'
}

# Create data frames with prediction results and error metrics
y_real = df_23["Power [MW]"]
metrics_data = []
for model, full_name in forecast_models.items():
    model_pred = df_23[f'{model} Predicted Power [MW]']
    MAE, MBE, MSE, RMSE, cvRMSE, NMBE = calculate_metrics(y_real, model_pred)
    metrics_data.append([full_name, MAE, MBE, MSE, RMSE, cvRMSE, NMBE])

df_metrics = pd.DataFrame(metrics_data, columns=['Methods', 'MAE', 'MBE', 'MSE', 'RMSE', 'cvMSE', 'NMBE'])

# Meteo features
meteo_features = ['Temperature [°C]', 'Solar Radiation [W/m2]', 'Wind Speed [m/s]', 'Rain [mm]' ]
initial_meteo_variable = meteo_features[0]

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define layout
app.layout = html.Div([
    html.H1('Paris Energy Forecast tool (MW)'),
    html.P('Representing Data and Forecasting for 2023'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Feature Selection', value='tab-3')
    ]),
    html.Div(id='tabs-content')
])

# Define content rendering callback
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        # Plot for power consumption data
        fig1 = px.line(df_20_23, x="Date", y=["Power [MW]"])
        power_graph = dcc.Graph(
            id='power-graph',
            figure=fig1
        )

        # Dropdown for meteorological data
        meteo_dropdown = dcc.Dropdown(
            id='meteo-dropdown',
            options=[{'label': col, 'value': col} for col in meteo_features],
            value=initial_meteo_variable
        )

        # Initial meteorological data plot
        initial_meteo_graph = dcc.Graph(
            id='meteo-graph'
        )

        return html.Div([
            html.H4('Paris consumption data'),
            power_graph,
            html.H4('Meteorological data'),
            meteo_dropdown,
            initial_meteo_graph
        ])

    elif tab == 'tab-2':
        return html.Div([
            html.H4('Select Forecast Model:'),
            dcc.Dropdown(
                id='model-dropdown',
                options=[{'label': full_name, 'value': model} for model, full_name in forecast_models.items()],
                value='Linear'
            ),
            dcc.Graph(id='forecast-plot'),
            html.H4('Forecast Model Comparison Metrics:'),
            generate_table(df_metrics)
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H4('Feature Selection'),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': col, 'value': col} for col in df_20_23.columns[1:]],  # Exclude 'Date'
                value=df_20_23.columns[1]  # Initialize with the first feature (excluding 'Date')
            ),
            dcc.Graph(id='feature-graph'),
            html.Div([
                html.H4('Performing feature selection'),
                dbc.Button("Correlation Matrix", id="correlation-matrix-button", color="primary", className="mr-2"),
                dbc.Button("Filter Method", id="filter-method-button", color="primary", className="mr-2"),
                dbc.Button("Wrapped Method", id="wrapped-method-button", color="primary", className="mr-2"),
                dbc.Button("Embedded Method", id="embedded-method-button", color="primary", className="mr-2"),
                dbc.Button("Final Feature Selection", id="final-feature-selection-button", color="primary", className="mr-2"),
            ]),
            html.Div(id='feature-sections-content')  # Placeholder for feature sections content
        ])

# Callback to update the plot based on the selected variable
@app.callback(
    Output('meteo-graph', 'figure'),
    [Input('meteo-dropdown', 'value')]
)
def update_meteo_graph(selected_variable):
    feature_colors = {
        'Temperature [°C]': 'coral',
        'Relative Humidity [%]': 'magenta',
        'Solar Radiation [W/m2]': 'orange',
        'Rain [mm]': 'darkorchid',
        'Wind Speed [m/s]':'silver'
    }
    fig_meteo = px.line(df_20_23, x='Date', y=selected_variable)
    # Set the color for the selected variable
    fig_meteo.update_traces(line=dict(color=feature_colors[selected_variable]))
    return fig_meteo

# Define callback to update the plot based on selected feature
@app.callback(
    Output('feature-graph', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_feature_graph(selected_feature):
    fig_feature = px.scatter(df_20_23, x='Date', y=selected_feature)
    return fig_feature

# Callback to render feature sections
@app.callback(
    Output('feature-sections-content', 'children'),
    Input('correlation-matrix-button', 'n_clicks'),
    Input('filter-method-button', 'n_clicks'),
    Input('wrapped-method-button', 'n_clicks'),
    Input('embedded-method-button', 'n_clicks'),
    Input('final-feature-selection-button', 'n_clicks')
)
def render_feature_sections(correlation_clicks, filter_clicks, wrapped_clicks, embedded_clicks, final_selection_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'correlation-matrix-button':
        # Calculate correlation matrix
        corr_sample = df_20_23.iloc[:, 1:].sample(frac=0.5).corr()

        # Plot correlation matrix using Plotly's heatmap
        fig = px.imshow(corr_sample, color_continuous_scale='RdBu_r', labels=dict(color="Correlation"),
                        x=corr_sample.index, y=corr_sample.columns,
                        zmin=-1, zmax=1)  # Set the color scale range from -1 to 1
        # Update layout
        fig.update_layout(title='Correlation Matrix',
                          width=1200, height=600,  # Adjust width and height as needed
                          xaxis=dict(tickangle=-90, tickfont=dict(size=12)),
                          yaxis=dict(tickfont=dict(size=12)))  # Rotate x-axis labels vertically

        # Convert to HTML div element
        correlation_div = dcc.Graph(id='correlation-heatmap', figure=fig)
        return correlation_div

    elif button_id == 'filter-method-button':
        # Define selected features and their scores for mutual_info_regression
        selected_features_mi = ['Power-1 [MW]', 'Temperature [°C]', 'CAC40', 'Solar Radiation [W/m2]',
                                'Month', 'Wind Speed [m/s]', 'Hour', 'DayOfWeek', 'Inhabitants [x 10E6]','Rain [mm]']
        scores_mi = [1.0000, 0.2366, 0.1696, 0.1569, 0.1532, 0.1143, 0.0923, 0.0136, 0.0091, 0.0089]

        # Define selected features and their scores for f_regression
        selected_features_f = ['Power-1 [MW]', 'Temperature [°C]', 'Hour', 'Wind Speed [m/s]', 'Month', 'DayOfWeek',
                                'Inhabitants [x 10E6]', 'CAC40', 'Solar Radiation [W/m2]', 'Rain [mm]']
        scores_f = [1.0000, 0.0044, 0.0006, 0.0005, 0.0002, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000]

        # Create bar plot for mutual_info_regression
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=scores_mi,
            y=selected_features_mi,
            orientation='h',
            marker=dict(color='darkturquoise'),
            name='mutual_info_regression'
        ))
        #Add bars for f_regression
        fig_bar.add_trace(go.Bar(
            x=scores_f,
            y=selected_features_f,
            orientation='h',
            marker=dict(color='salmon'),
            name='f_regression'
        ))
        fig_bar.update_layout(title='Feature Scores - Mutual Info Regression & F Regression',
                              xaxis_title='Score',
                              height=500)  # Adjust height as needed
        # Convert to HTML div element
        filter_div = dcc.Graph(id='feature-scores-bar', figure=fig_bar)
        return filter_div

    elif button_id == 'wrapped-method-button':
        # Define selected features and their scores for Forward Selection
        selected_features_forward = ['Power-1 [MW]','Temperature [°C]','DayOfWeek', 'Solar Radiation [W/m2]',
                                     'Inhabitants [x 10E6]', 'Month', 'Rain [mm]', 'Wind Speed [m/s]',
                                     'CAC40', 'Hour']
        mean_r2_scores_forward = [0.9953, 0.9953, 0.9953, 0.9953, 0.9953, 0.9953, 0.9953, 0.9953, 0.9953, 0.9953]
        scaler = MinMaxScaler()
        # Reshape the scores to fit the scaler
        mean_r2_scores_array = np.array(mean_r2_scores_forward).reshape(-1, 1)
        # Fit and transform the scores using MinMaxScaler
        scaled_scores = scaler.fit_transform(mean_r2_scores_array)
        # Invert the scaled scores
        scaled_scores = 1 - scaled_scores
        # Convert the scaled scores back to a list
        scaled_scores_list = scaled_scores.flatten().tolist()

        # Create bar plot for Forward Selection with scaled scores
        fig_bar_forward_scaled = go.Figure(go.Bar(
        x=scaled_scores_list,
        y=selected_features_forward,
        orientation='h',
        marker=dict(color='darkorchid'),
        name='Forward Selection (Scaled)'
        ))

        fig_bar_forward_scaled.update_layout(
            title='Feature Scores - Forward Selection (Scaled)',
            xaxis_title='1 - scaled score (MinMaxScaled)',
            height=500
        )
        # Define RFE rankings
        rfe_rankings = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]

        rfe_features = ['Power-1 [kW]', 'Hour', 'Temperature [°C]', 'Solar Radiation [W/m2]', 'CAC40',
                        'Wind Speed [m/s]', 'DayOfWeek', 'Month', 'Inhabitants [x 10E6]', 'Rain [mm]']
        # Combine features and rankings into a list of dictionaries
        rfe_data = [{'Ranking': rank, 'Feature': feature} for rank, feature in zip(rfe_rankings, rfe_features)]

        # Create Dash DataTable for the RFE rankings
        rfe_table = dash_table.DataTable(
            id='rfe-ranking-table',
            columns=[
                {'name': 'Ranking', 'id': 'Ranking'},
                {'name': 'Feature', 'id': 'Feature'}
            ],
            data=rfe_data,
            style_table={'height': '300px', 'overflowY': 'auto'}
        )
        # Return the bar plot for forward selection along with the RFE rankings table
        return html.Div([
            dcc.Graph(id='feature-scores-bar-scaled', figure=fig_bar_forward_scaled),
            html.H4('RFE Rankings'),
            rfe_table
        ])

    elif button_id == 'embedded-method-button':
        # Define feature importances
        feature_importances = [
            {'Feature': 'Power-1 [kW]', 'Importance': 0.996143},
            {'Feature': 'Hour', 'Importance': 0.002386},
            {'Feature': 'Solar Radiation [W/m2]', 'Importance': 0.000357},
            {'Feature': 'Temperature [°C]', 'Importance':  0.000308},
            {'Feature': 'CAC40', 'Importance': 0.000227},
            {'Feature': 'Wind Speed [m/s]', 'Importance':0.000220},
            {'Feature': 'DayOfWeek', 'Importance': 0.000179},
            {'Feature': 'Month', 'Importance': 0.000092},
            {'Feature': 'Inhabitants [x 10E6]', 'Importance': 0.000047},
            {'Feature': 'Rain [mm]', 'Importance': 0.000040}
        ]
        # Scale the feature importances
        scaler = MinMaxScaler()
        importance_values = [entry['Importance'] for entry in feature_importances]
        scaled_importances = scaler.fit_transform(np.array(importance_values).reshape(-1, 1))
        # Update the feature importances with scaled values

        for i, entry in enumerate(feature_importances):
            entry['Importance'] = scaled_importances[i][0]

        # Sort the feature importances by importance value
        feature_importances.sort(key=lambda x: x['Importance'], reverse=True)
        # Create a bar plot for feature importances
        fig_bar_embedded_scaled = go.Figure(go.Bar(
            x=[entry['Importance'] for entry in feature_importances],
            y=[entry['Feature'] for entry in feature_importances],
            orientation='h',
            marker=dict(color='mediumseagreen'),
            name='Embedded Method (Scaled)'
        ))

        fig_bar_embedded_scaled.update_layout(
            title='Feature Importances - Embedded Method (Scaled)',
            xaxis_title='Scaled Importance',
            height=500
        )
        # Convert to HTML div element
        embedded_div = dcc.Graph(id='feature-importance-bar-scaled', figure=fig_bar_embedded_scaled)
        return embedded_div

    elif button_id == 'final-feature-selection-button':
        # Define final feature ranking
        final_feature_ranking = [
            {'Variable': 'Power-1 [kW]', 'Rank': 1, 'Average Rank': 1.00},
            {'Variable': 'Temperature [°C]', 'Rank': 2, 'Average Rank': 2.60},
            {'Variable': 'Solar Radiation [W/m2]', 'Rank': 3, 'Average Rank': 4.80},
            {'Variable': 'Hour', 'Rank': 4, 'Average Rank': 4.80},
            {'Variable': 'Wind Speed [m/s]', 'Rank': 5, 'Average Rank': 6.00},
            {'Variable': 'CAC40', 'Rank': 6, 'Average Rank': 6.00},
            {'Variable': 'DayOfWeek', 'Rank': 7, 'Average Rank': 6.20},
            {'Variable': 'Month', 'Rank': 8, 'Average Rank': 6.40},
            {'Variable': 'Inhabitants [x 10E6]', 'Rank': 9, 'Average Rank': 7.80},
            {'Variable': 'Rain [mm]', 'Rank': 10, 'Average Rank': 9.40}

        ]
        # Create Dash DataTable for the final feature ranking
        final_feature_table = dash_table.DataTable(
            id='final-feature-ranking-table',
            columns=[
                {'name': 'Variable', 'id': 'Variable'},
                {'name': 'Rank', 'id': 'Rank'},
                {'name': 'Average Rank', 'id': 'Average Rank'},
            ],
            data=final_feature_ranking,
            style_data_conditional=[
                {
                    'if': {'row_index': i},
                    'backgroundColor': '#DCF8C6',  # Lighter green color
                    'color': 'black'
                } for i in range(6)
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_table={'height': '400px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'center'}
        )

        # Return the final feature ranking table
        return html.Div([
            html.H4('Final Feature Ranking'),
            final_feature_table,
            html.P('Light green is for selected features'),
        ])

# Define callback to update the plot based on selected forecast model
@app.callback(
    Output('forecast-plot', 'figure'),
    Input('model-dropdown', 'value')
)
def update_forecast_plot(selected_model):
    model_pred_column = f'{selected_model} Predicted Power [MW]'
    fig = px.line(df_23, x='Date', y=[df_23['Power [MW]'], df_23[model_pred_column]], labels={'value': 'Power [kW]'}, title=f'{forecast_models[selected_model]} Forecast')
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)

