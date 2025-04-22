import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import plotly.graph_objects as go

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback_context

# ==================================================================
# Завантаження та підготовка даних
# ==================================================================
DATA_DIR = "data"

def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, filename))

orders = load_data("liqpay_order.csv")
orders["create_date"] = pd.to_datetime(orders["create_date"], errors="coerce")

# агрегуємо фічі на рівні користувача
today = datetime.today()
user_feats = orders.groupby("user_id").agg(
    total_amount=("amount", "sum"),
    count=("amount", "count"),
    last_date=("create_date", "max"),
    first_date=("create_date", "min")
).reset_index()

user_feats["avg_amount"] = user_feats["total_amount"] / user_feats["count"]
user_feats["recency"] = (today - user_feats["last_date"]).dt.days
# місяці активності
user_feats["months_active"] = (
    (user_feats["last_date"].dt.year - user_feats["first_date"].dt.year) * 12 +
    (user_feats["last_date"].dt.month - user_feats["first_date"].dt.month) + 1
).clip(lower=1)
user_feats["frequency"] = user_feats["count"] / user_feats["months_active"]
user_feats["churn"] = (user_feats["recency"] > 90).astype(int)
user_feats.dropna(subset=["total_amount","avg_amount","recency","frequency"], inplace=True)

FEATURES = ["total_amount","avg_amount","recency","frequency"]
TARGET = "churn"
X = user_feats[FEATURES]
y = user_feats[TARGET]

# ==================================================================
# Навчання моделі
# ==================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

user_feats["churn_proba"] = model.predict_proba(user_feats[FEATURES])[:, 1]
user_feats["churn_pred"] = model.predict(user_feats[FEATURES])

# ==================================================================
# ROC крива та важливість фіч
# ==================================================================
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC крива'))
roc_fig.update_layout(
    title=f"ROC крива (AUC={roc_auc_score(y_test, y_proba):.2f})",
    xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
    template='plotly_white', margin=dict(l=40, r=40, t=60, b=40)
)

importances = model.feature_importances_
fi_fig = go.Figure(go.Bar(x=FEATURES, y=importances))
fi_fig.update_layout(
    title='Важливість фіч', xaxis_title='Фічі', yaxis_title='Вага',
    template='plotly_white', margin=dict(l=40, r=40, t=60, b=40)
)

# ==================================================================
# Інтерфейс Dash для вкладки Churn
# ==================================================================
def render_churn_tab():
    return dbc.Container([
        dbc.Row(dbc.Col(html.H2("Прогноз відтоку донорів", className="mt-4 mb-3 text-center text-primary"))),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Фільтри"),
                dbc.CardBody([
                    html.Label("Ймовірність відтоку >=:"),
                    dcc.Slider(
                        id="churn-proba-slider", min=0, max=1, step=0.01, value=0.5,
                        marks={i/10: f"{int(i*10)}%" for i in range(11)}, tooltip={'placement': 'bottom'}
                    ),
                    html.Br(),
                    html.Label("Бінів на гістограмі:"),
                    dcc.Slider(
                        id="hist-bins-slider", min=5, max=50, step=1, value=20,
                        marks={i: str(i) for i in range(5, 51, 5)}, tooltip={'placement': 'bottom'}
                    ),
                    html.Br(),
                    html.Label("Прогнозована відмітка:"),
                    dcc.RadioItems(
                        id="churn-pred-filter",
                        options=[
                            {"label": "Усі", "value": "all"},
                            {"label": "Відтік (1)", "value": 1},
                            {"label": "Ні (0)",  "value": 0}
                        ], value="all",
                        inline=True
                    ),
                ])
            ], className="mb-4"), width=4),
            dbc.Col(dcc.Graph(id="churn-proba-hist", config={'displayModeBar': False}), width=8)
        ]),
        dbc.Row(dbc.Col(html.H4("Таблиця прогнозів", className="mt-4 mb-2 text-secondary"))),
        dbc.Row(dbc.Col(dbc.Button("Експорт CSV", id="download-button", color="info", className="mb-2"))),
        dbc.Row(dbc.Col(dcc.Download(id="download-churn-data"))),
        dbc.Row(dbc.Col(dash_table.DataTable(
            id="churn-table",
            columns=[{"name": c, "id": c} for c in ["user_id","total_amount","avg_amount","recency","frequency","churn_proba","churn_pred"]],
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': '#e9ecef', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'center', 'padding': '5px'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}],
            page_size=10
        ), width=12)),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("ROC крива"), dbc.CardBody(dcc.Graph(id="churn-roc-curve", figure=roc_fig, config={'displayModeBar': False}))]), width=6),
            dbc.Col(dbc.Card([dbc.CardHeader("Важливість фіч"), dbc.CardBody(dcc.Graph(id="churn-feature-imp", figure=fi_fig, config={'displayModeBar': False}))]), width=6)
        ], className="mt-4 mb-4")
    ], fluid=True)


def register_churn_callbacks(app: dash.Dash):
    @app.callback(
        [Output("churn-proba-hist", "figure"),
         Output("churn-table", "data"),
         Output("download-churn-data", "data")],
        [Input("churn-proba-slider", "value"),
         Input("hist-bins-slider", "value"),
         Input("churn-pred-filter", "value"),
         Input("download-button", "n_clicks")]
    )
    def update_churn_outputs(threshold, bins, pred_filter, n_clicks):
        # визначаємо, що саме спрацювало
        triggered = callback_context.triggered[0]['prop_id'].split('.')[0]
        df = user_feats.copy()
        df = df[df["churn_proba"] >= threshold]
        if pred_filter in (0, 1):
            df = df[df["churn_pred"] == pred_filter]

        # оновлена гістограма
        hist = go.Figure(go.Histogram(x=df["churn_proba"], nbinsx=bins))
        hist.update_layout(
            title=f"Ймовірності відтоку >= {threshold:.2f}",
            xaxis_title="Ймовірність відтоку", yaxis_title="Кількість", template='plotly_white',
            margin=dict(l=40,r=40,t=60,b=40)
        )
        # дані для таблиці
        table_data = df.to_dict('records')

        # формуємо CSV лише при кліку на кнопку
        download = None
        if triggered == 'download-button' and n_clicks:
            download = dcc.send_data_frame(df.to_csv, filename="churn_data.csv", index=False)

        return hist, table_data, download
