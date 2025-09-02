import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import ast
import plotly.graph_objs as go

# -------------------- Dash App Setup --------------------
app = dash.Dash(__name__)
server = app.server
log_file = "data/emotion_log.csv"

emotions_list = ['happy','sad','angry','surprise','fear','disgust','neutral']

app.layout = html.Div([
    html.H1("Real-Time Emotion & Intensity Dashboard"),
    dcc.Graph(id="emotion-trend"),
    html.Div(id="alerts-div"),
    dcc.Interval(id="interval-component", interval=1000, n_intervals=0)
])

@app.callback(
    [Output("emotion-trend", "figure"),
     Output("alerts-div", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_graph(n):
    try:
        df = pd.read_csv(log_file)
        if df.empty:
            return {}, ""

        # Parse intensity scores
        df['Intensity Scores'] = df['Intensity Scores'].apply(ast.literal_eval)

        fig = go.Figure()
        for emo in emotions_list:
            fig.add_trace(go.Scatter(
                y=df['Intensity Scores'].apply(lambda x: x.get(emo,0)),
                mode='lines+markers',
                name=emo
            ))
        fig.update_layout(title="Emotion Intensity Trend", yaxis=dict(range=[0,100]))

        # Current alerts
        alerts = df['Alerts'].iloc[-1] if not df['Alerts'].empty else ""
        return fig, html.Div([html.H3("Current Alerts:"), html.P(alerts)])
    except Exception as e:
        return {}, f"Error: {e}"

if __name__ == "__main__":
    app.run_server(debug=False)
