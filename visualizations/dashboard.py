import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn import metrics
from datetime import datetime
from sqlalchemy import create_engine
import pickle

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "happiness_dw"
}

# Load PCA model information
try:
    with open("./models/happiness_score_pca_model.pkl", "rb") as f:
        model_package = pickle.load(f)
    pca = model_package["pca"]
    n_components = pca.n_components_
    print(f"PCA Model loaded: {n_components} components")
except Exception as e:
    print(f"Could not load PCA model: {e}")
    n_components = 3

def load_predictions_from_db(filter_type='test'):
    try:
        engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        )
        query = """
            SELECT id, gdp_per_capita, family, freedom, life_expectancy, 
                   social_support, y_real, y_pred, is_train 
            FROM happiness_predictions
        """
        if filter_type == 'train':
            query += " WHERE is_train = 1"
        elif filter_type == 'test':
            query += " WHERE is_train = 0"
        query += " ORDER BY id;"
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_metrics(df):
    if len(df) == 0:
        return {
            'r2': 0, 'mae': 0, 'rmse': 0, 'mape': 0,
            'df': pd.DataFrame(), 'total_predictions': 0,
            'mean_error': 0, 'std_error': 0
        }
    y_real = df['y_real']
    y_pred = df['y_pred']
    r2 = metrics.r2_score(y_real, y_pred)
    mae = metrics.mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_real, y_pred))
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    df['error'] = df['y_real'] - df['y_pred']
    df['abs_error'] = np.abs(df['error'])
    df['pct_error'] = (df['abs_error'] / df['y_real']) * 100
    mean_error = df['error'].mean()
    std_error = df['error'].std()
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'df': df,
        'total_predictions': len(df),
        'mean_error': mean_error,
        'std_error': std_error
    }

# Dash App
app = dash.Dash(__name__, suppress_callback_exceptions=True, meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])
app.title = "ML Model Monitor"

COLORS = {
    'background': '#0a0e27',
    'card': '#151932',
    'card_hover': '#1a1f3a',
    'text': '#e8eaed',
    'text_secondary': '#9aa0a6',
    'primary': '#4c8bf5',
    'success': '#34a853',
    'warning': '#fbbc04',
    'danger': '#ea4335',
    'info': '#a142f4',
    'accent': '#ff6b6b'
}

# Styles
CARD_STYLE = {
    'backgroundColor': COLORS['card'],
    'padding': '25px',
    'borderRadius': '12px',
    'marginBottom': '20px',
    'boxShadow': '0 4px 15px rgba(0, 0, 0, 0.5)',
    'border': f'1px solid {COLORS["card_hover"]}',
    'transition': 'all 0.3s ease'
}

KPI_CARD_STYLE = {
    **CARD_STYLE,
    'textAlign': 'center',
    'height': '140px',
    'display': 'flex',
    'flexDirection': 'column',
    'justifyContent': 'center',
    'alignItems': 'center',
    'padding': '20px'
}


# Layout
app.layout = html.Div(style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '30px 40px', 'fontFamily': 'Arial, sans-serif'}, children=[
    
    html.Div(style={'background': f'linear-gradient(135deg, {COLORS["primary"]}, {COLORS["info"]})', 'padding': '30px', 'borderRadius': '15px', 'marginBottom': '30px', 'boxShadow': '0 8px 20px rgba(0, 0, 0, 0.4)'}, children=[
        html.H1('ML Model Performance Monitor', style={'color': 'white', 'marginBottom': '8px', 'fontSize': '32px', 'fontWeight': '700'}),
        html.P(f'PCA-based Happiness Score Prediction ({n_components} components)', style={'color': 'rgba(255,255,255,0.9)', 'fontSize': '16px', 'margin': '0'})
    ]),
    
    html.Div(style={**CARD_STYLE, 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '15px'}, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '15px'}, children=[
            html.Label('Data Filter:', style={'color': COLORS['text'], 'fontSize': '15px', 'fontWeight': '600'}),
            dcc.Dropdown(
                id='data-filter',
                options=[
                    {'label': 'Training Data', 'value': 'train'},
                    {'label': 'Test Data', 'value': 'test'},
                    {'label': 'All Data', 'value': 'all'}
                ],
                value='test',
                clearable=False,
                style={
                    'width': '220px',
                    'color': COLORS['text'],
                    'backgroundColor': COLORS['card_hover']
                },
                className="dark-dropdown"
            )
        ]),
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '20px'}, children=[
            html.Div(id='status-indicator', children=[
                html.Span('●', style={'color': COLORS['success'], 'fontSize': '20px', 'marginRight': '8px'}),
                html.Span('Live', style={'color': COLORS['text_secondary'], 'fontSize': '14px'})
            ]),
            html.Span(id='last-update', style={'color': COLORS['text_secondary'], 'fontSize': '14px'}),
            html.Span(id='record-count', style={'backgroundColor': COLORS['primary'], 'color': 'white', 'padding': '6px 14px', 'borderRadius': '20px', 'fontSize': '13px', 'fontWeight': '600'})
        ])
    ]),
    
    html.Div(id='kpi-section', style={'marginBottom': '25px'}),
    
    html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(550px, 1fr))', 'gap': '20px', 'marginBottom': '20px'}, children=[
        html.Div(style=CARD_STYLE, children=[
            html.H3('Predicted vs Actual Values', style={'color': COLORS['text'], 'margin': '0', 'fontSize': '18px', 'fontWeight': '600'}),
            dcc.Graph(id='scatter-predictions', style={'height': '400px'}, config={'displayModeBar': False})
        ]),
        html.Div(style=CARD_STYLE, children=[
            html.H3('Prediction Error Distribution', style={'color': COLORS['text'], 'margin': '0', 'fontSize': '18px', 'fontWeight': '600'}),
            dcc.Graph(id='error-distribution', style={'height': '400px'}, config={'displayModeBar': False})
        ]),
    ]),
    
    html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(550px, 1fr))', 'gap': '20px', 'marginBottom': '20px'}, children=[
        html.Div(style=CARD_STYLE, children=[
            html.H3('Residual Analysis', style={'color': COLORS['text'], 'margin': '0', 'fontSize': '18px', 'fontWeight': '600'}),
            dcc.Graph(id='residuals-plot', style={'height': '400px'}, config={'displayModeBar': False})
        ]),
        html.Div(style=CARD_STYLE, children=[
            html.H3('Error by Prediction Range', style={'color': COLORS['text'], 'margin': '0', 'fontSize': '18px', 'fontWeight': '600'}),
            dcc.Graph(id='error-by-range', style={'height': '400px'}, config={'displayModeBar': False})
        ]),
    ]),
    
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0),
    dcc.Store(id='data-store')
])

app.index_string = app.index_string.replace(
    "</head>",
    """
    <style>
    .Select-menu-outer {
        background-color: #1a1f3a !important;
        color: #e8eaed !important;
    }
    .Select-value-label {
        color: #e8eaed !important;
    }
    .Select-control {
        background-color: #1a1f3a !important;
        border-color: #3c4048 !important;
    }
    </style>
    </head>
    """
)

# Callbacks
@callback(
    [Output('data-store', 'data'),
     Output('last-update', 'children'),
     Output('record-count', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('data-filter', 'value')]
)
def update_data(n_intervals, filter_type):
    df = load_predictions_from_db(filter_type)
    timestamp = datetime.now().strftime('%H:%M:%S')
    if len(df) == 0:
        return {}, f"{timestamp}", "0 records"
    filter_labels = {'train': 'Training', 'test': 'Test', 'all': 'All'}
    return df.to_dict('records'), timestamp, f"{len(df)} {filter_labels.get(filter_type, '')} records"

@callback(
    [Output('scatter-predictions', 'figure'),
     Output('error-distribution', 'figure'),
     Output('residuals-plot', 'figure'),
     Output('error-by-range', 'figure'),
     Output('kpi-section', 'children')],
    Input('data-store', 'data')
)
def update_graphs(data):
    if not data:
        empty_fig = go.Figure()
        empty_fig.update_layout(template='plotly_dark')
        empty_kpi = html.Div("Waiting for data...", style={'textAlign': 'center', 'padding': '40px'})
        return [empty_fig]*7 + [empty_kpi]
    
    df = pd.DataFrame(data)
    metrics_data = calculate_metrics(df)
    df_plot = metrics_data['df']

    # Scatter Predictions vs Real
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(x=df_plot['y_real'], y=df_plot['y_pred'], mode='markers'))
    scatter_fig.update_layout(template='plotly_dark', xaxis_title='Actual', yaxis_title='Predicted')

    # Error Distribution
    error_dist = go.Figure()
    error_dist.add_trace(go.Histogram(x=df_plot['error']))
    error_dist.update_layout(template='plotly_dark', xaxis_title='Error', yaxis_title='Frequency')

    # Residuals Plot
    residuals_plot = go.Figure()
    residuals_plot.add_trace(go.Scatter(x=df_plot['y_pred'], y=df_plot['error'], mode='markers'))
    residuals_plot.update_layout(template='plotly_dark', xaxis_title='Predicted', yaxis_title='Residual')

    # Error by Prediction Range
    df_plot['pred_bin'] = pd.cut(df_plot['y_pred'], bins=10)
    error_by_bin = df_plot.groupby('pred_bin', observed=True).agg({'abs_error': 'mean', 'y_pred': 'count'}).reset_index()
    error_range_fig = go.Figure()
    error_range_fig.add_trace(go.Bar(x=error_by_bin['pred_bin'].astype(str), y=error_by_bin['abs_error']))
    error_range_fig.update_layout(template='plotly_dark', xaxis_title='Prediction Range', yaxis_title='Mean Absolute Error')

    # KPI Section
    kpi_data = [
        ('R² Score', f"{metrics_data['r2']:.4f}"),
        ('MAE', f"{metrics_data['mae']:.4f}"),
        ('RMSE', f"{metrics_data['rmse']:.4f}"),
        ('MAPE', f"{metrics_data['mape']:.2f}%"),
        ('Mean Error', f"{metrics_data['mean_error']:.4f}"),
        ('Std Error', f"{metrics_data['std_error']:.4f}")
    ]
    kpi_section = html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(180px, 1fr))', 'gap': '15px'}, children=[
        html.Div(style=KPI_CARD_STYLE, children=[
            html.P(label, style={'color': COLORS['text_secondary'], 'fontSize': '13px'}),
            html.H2(value, style={'color': COLORS['primary'], 'fontSize': '28px'})
        ]) for label, value in kpi_data
    ])

    return [scatter_fig, error_dist, residuals_plot, error_range_fig, kpi_section]

if __name__ == '__main__':
    app.run(debug=True, port=8050)