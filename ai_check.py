import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, roc_curve
import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback_context

# ==================================================================
# Стилі та палітра
# ==================================================================
OLIVE = '#556b2f'
OLIVE_LIGHT = '#829661'
TEMPLATE = 'plotly_white'

# ==================================================================
# Завантаження та підготовка даних
# ==================================================================
DATA_DIR = "data"

def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, filename))

orders = load_data("liqpay_order.csv")
orders["create_date"] = pd.to_datetime(orders["create_date"], errors="coerce")

user_feats = (
    orders.sort_values('create_date')
    .groupby("user_id")
    .agg(
        total_amount=("amount", "sum"),
        count=("amount", "count"),
        last_amount=("amount", lambda x: x.iloc[-1]),
        last_date=("create_date", "max"),
        first_date=("create_date", "min")
    )
    .reset_index()
)
today = datetime.today()
user_feats["avg_amount"] = user_feats["total_amount"] / user_feats["count"]
user_feats["recency"] = (today - user_feats["last_date"]).dt.days
user_feats["months_active"] = (
    (user_feats["last_date"].dt.year - user_feats["first_date"].dt.year) * 12 +
    (user_feats["last_date"].dt.month - user_feats["first_date"].dt.month) + 1
).clip(lower=1)
user_feats["frequency"] = user_feats["count"] / user_feats["months_active"]
user_feats["churn"] = (user_feats["recency"] > 90).astype(int)
user_feats.dropna(subset=["avg_amount","recency","frequency","last_amount"], inplace=True)

# ==================================================================
# Навчання моделей
# ==================================================================
FEATURES = ["avg_amount","recency","frequency"]

# Класифікація відтоку
Xc = user_feats[FEATURES]
yc = user_feats["churn"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.3, random_state=42, stratify=yc)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(Xc_train, yc_train)
user_feats["churn_proba"] = clf.predict_proba(user_feats[FEATURES])[:,1]
user_feats["churn_pred"] = clf.predict(user_feats[FEATURES])

# Регресія рекомендацій
Xr = user_feats[FEATURES]
yr = user_feats["last_amount"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.3, random_state=42)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(Xr_train, yr_train)
user_feats["next_best_amount"] = reg.predict(user_feats[FEATURES])

# KPI
avg_churn = user_feats["churn_proba"].mean()
perc_churn = user_feats["churn_pred"].mean() * 100
avg_next = user_feats["next_best_amount"].mean()

# ==================================================================
# Функція створення графіків за замовчуванням з українськими підписами
# ==================================================================
def create_default_figures():
    fpr, tpr, _ = roc_curve(yc_test, clf.predict_proba(Xc_test)[:,1])
    roc_fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode='lines', line_color=OLIVE))
    roc_fig.update_layout(
        title=f"Крива ROC (AUC={roc_auc_score(yc_test, clf.predict_proba(Xc_test)[:,1]):.2f})",
        xaxis_title='Частка хибних спрацьовувань', yaxis_title='Чутливість', template=TEMPLATE
    )
    fi_fig = px.bar(
        x=FEATURES, y=clf.feature_importances_,
        labels={'x':'Характеристика','y':'Вага характеристики'},
        title='Важливість характеристик моделі відтоку', color_discrete_sequence=[OLIVE], template=TEMPLATE
    )
    hist_fig = px.histogram(
        user_feats, x="next_best_amount", nbins=20,
        title="Розподіл рекомендованих сум для наступного донату",
        labels={'next_best_amount':'Сума (грн)'}, color_discrete_sequence=[OLIVE_LIGHT], template=TEMPLATE
    )
    scatter_fig = px.scatter(
        user_feats, x="churn_proba", y="next_best_amount",
        color="churn_pred", symbol="churn_pred",
        labels={'churn_proba':'Ймовірність відтоку','next_best_amount':'Рекомендована сума (грн)'},
        title="Ймовірність відтоку vs Рекомендована сума", color_discrete_sequence=[OLIVE], template=TEMPLATE
    )
    box_fig = px.box(
        user_feats, x="churn_pred", y="next_best_amount",
        labels={'churn_pred':'Прогноз відтоку','next_best_amount':'Рекомендована сума (грн)'},
        title="Порівняння рекомендованих сум: відтік vs не відтік",
        color_discrete_sequence=[OLIVE_LIGHT, OLIVE], template=TEMPLATE
    )
    rec_hist = px.histogram(
        user_feats, x='recency', nbins=20,
        labels={'recency':'Дні з останнього донату'},
        title='Розподіл часу від останнього донату', color_discrete_sequence=[OLIVE], template=TEMPLATE
    )
    top10_fig = px.bar(
        user_feats.nlargest(10,'next_best_amount'), x='user_id', y='next_best_amount',
        labels={'user_id':'ID користувача','next_best_amount':'Сума (грн)'},
        title='Топ 10 користувачів за рекомендованою сумою', color_discrete_sequence=[OLIVE], template=TEMPLATE
    )
    return roc_fig, fi_fig, hist_fig, scatter_fig, box_fig, rec_hist, top10_fig

roc_default, fi_default, nb_default, scatter_default, box_default, recency_default, top10_default = create_default_figures()

# ==================================================================
# Інтерфейс вкладки з українськими підписами та коротким описом графіків
# ==================================================================
def render_churn_tab():
    min_date = user_feats['last_date'].min().date()
    max_date = user_feats['last_date'].max().date()
    return dbc.Container([
        # KPI Cards
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Середня ймовірність відтоку"), html.H4(f"{avg_churn:.2f}")]), color='light'), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("% користувачів з відтоком"), html.H4(f"{perc_churn:.1f}%")]), color='light'), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Середня рекомендована сума (грн)"), html.H4(f"{avg_next:.2f}")]), color='light'), width=4)
        ], className="mt-4 mb-4"),

        # Фільтри даних
        dbc.Card([
            dbc.CardHeader("Фільтри"),
            dbc.CardBody([
                html.Label("Дата останнього донату:"),
                dcc.DatePickerRange(id='date-range', start_date=min_date, end_date=max_date, display_format='YYYY-MM-DD'),
                html.Br(), html.Br(),
                html.Label("Ймовірність відтоку >=:"),
                dcc.Slider(id="churn-proba-slider", min=0, max=1, step=0.01, value=0.5, marks={i/10:f"{i*10}%" for i in range(11)}),
                html.Br(),
                html.Label("Рекомендована сума >= (грн):"),
                dcc.Input(id="amount-threshold", type="number", value=0, placeholder="від"),
                html.Br(), html.Br(),
                html.Label("Статус відтоку:"),
                dcc.RadioItems(id="churn-pred-filter", options=[{'label':'Усі','value':'all'},{'label':'Відтік','value':1},{'label':'Не відтік','value':0}], value='all', inline=True)
            ])
        ], color='light', className="mb-4"),

        # Основні графіки
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Крива ROC"), dbc.CardBody(dcc.Graph(id="churn-roc-curve", figure=roc_default))], color='light'), width=6),
            dbc.Col(dbc.Card([dbc.CardHeader("Важливість характеристик"), dbc.CardBody(dcc.Graph(id="churn-feature-imp", figure=fi_default))], color='light'), width=6)
        ], className="mb-4"),

        # Додаткові графіки
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Гістограма рекомендованих сум"), dbc.CardBody(dcc.Graph(id="churn-proba-hist", figure=nb_default))], color='light'), width=6),
            dbc.Col(dbc.Card([dbc.CardHeader("Ймовірність відтоку vs Рекомендована сума"), dbc.CardBody(dcc.Graph(id="churn-scatter", figure=scatter_default))], color='light'), width=6)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Boxplot рекомендацій за статусом відтоку"), dbc.CardBody(dcc.Graph(id='churn-box', figure=box_default))], color='light'), width=6),
            dbc.Col(dbc.Card([dbc.CardHeader("Розподіл часу з останнього донату"), dbc.CardBody(dcc.Graph(id='recency-hist', figure=recency_default))], color='light'), width=6)
        ], className="mb-4"),

        # Top 10 рекомендацій над таблицею
        dbc.Card([dbc.CardHeader("Top 10 користувачів за рекомендованою сумою"), dbc.CardBody(dcc.Graph(id='top10-bar', figure=top10_default))], color='light', className="mb-4"),

        # Таблиця даних
        dbc.Row(dbc.Col(html.H4("Дані прогнозу та рекомендацій"), width=12)),
        dbc.Row([dbc.Col(dbc.Button("Завантажити CSV", id="download-table", color="success", className="mb-2"), width=2), dbc.Col(dcc.Download(id="download-data"))]),
        dbc.Row(dbc.Col(dash_table.DataTable(
            id="churn-table",
            columns=[{"name":c,"id":c} for c in ["user_id","last_date","churn_proba","churn_pred","next_best_amount"]],
            page_size=10, filter_action='native', sort_action='native',
            style_table={'overflowX':'auto'}, style_header={'backgroundColor':OLIVE,'color':'#fff'}, style_cell={'textAlign':'center','padding':'5px'},
            style_data_conditional=[{'if':{'row_index':'odd'},'backgroundColor':'#f9f9f9'},{'if':{'filter_query':'{churn_pred} = 1'},'backgroundColor':'#ffe6e6'}]
        ), width=12)),

        # Опис графіків
        dbc.Row(dbc.Col(html.Div([
            html.H5("Опис візуалізацій:"),
            html.Ul([
                html.Li(html.B("Крива ROC:"), " показує здатність моделі відтоку розрізняти між класами (чутливість vs специфічність)."),
                html.Li(html.B("Важливість характеристик:"), " відображає, які фічі (середній донат, recency, частота) найбільше впливають на прогноз відтоку."),
                html.Li(html.B("Гістограма рекомендованих сум:"), " показує розподіл суми, яку модель рекомендує для наступного донату."),
                html.Li(html.B("Ймовірність відтоку vs Рекомендована сума:"), " досліджує зв'язок між ймовірністю відтоку та розміром рекомендації."),
                html.Li(html.B("Boxplot рекомендацій за статусом відтоку:"), " порівнює рекомендовані суми для користувачів, яких модель позначила як відтік vs не відтік."),
                html.Li(html.B("Розподіл часу з останнього донату:"), " показує, скільки днів минуло з останнього донату для кожного користувача."),
                html.Li(html.B("Top 10 користувачів за рекомендованою сумою:"), " виводить найвищі рекомендації сум для топ-10 користувачів.")
            ])
        ], style={'padding':'15px', 'backgroundColor':'#f8f9fa', 'borderRadius':'5px'}), width=12)),

    ], fluid=True)

# Callbacks

def register_churn_callbacks(app: dash.Dash):
    @app.callback(
        [Output("churn-roc-curve","figure"), Output("churn-feature-imp","figure"), Output("churn-proba-hist","figure"), Output("churn-scatter","figure"), Output('churn-box','figure'), Output('recency-hist','figure'), Output('churn-table','data'), Output('top10-bar','figure')],
        [Input("date-range","start_date"), Input("date-range","end_date"), Input("churn-proba-slider","value"), Input("amount-threshold","value"), Input("churn-pred-filter","value")]
    )
    def update_charts(start_date, end_date, thresh, amt_thresh, pred_filter):
        df = user_feats.copy()
        df = df[(df['last_date'] >= pd.to_datetime(start_date)) & (df['last_date'] <= pd.to_datetime(end_date))]
        df = df[df["churn_proba"]>=thresh]
        df = df[df["next_best_amount"]>=amt_thresh]
        if pred_filter in (0,1): df = df[df["churn_pred"]==pred_filter]
        if df.empty:
            return roc_default, fi_default, nb_default, scatter_default, box_default, recency_default, [], top10_default
        if df['churn'].nunique() < 2:
            roc = go.Figure()
            roc.add_annotation(text="Недостатньо даних для побудови ROC", xref="paper", yref="paper", showarrow=False)
            roc.update_layout(template=TEMPLATE)
        else:
            vals = roc_curve(df['churn'], clf.predict_proba(df[FEATURES])[:,1])
            roc = go.Figure(go.Scatter(x=vals[0], y=vals[1], mode='lines', line_color=OLIVE))
            roc.update_layout(title="Крива ROC", xaxis_title='Частка хибних спрацьовувань', yaxis_title='Чутливість', template=TEMPLATE)
        fi = fi_default
        hist = px.histogram(df, x="next_best_amount", nbins=20, labels={'next_best_amount':'Сума (грн)'}, title="Гістограма рекомендованих сум", color_discrete_sequence=[OLIVE_LIGHT], template=TEMPLATE)
        scat = px.scatter(df, x="churn_proba", y="next_best_amount", color="churn_pred", symbol="churn_pred", labels={'churn_proba':'Ймовірність відтоку','next_best_amount':'Сума (грн)','churn_pred':'Прогноз відтоку'}, title="Ймовірність відтоку vs Рекомендована сума", color_discrete_sequence=[OLIVE], template=TEMPLATE)
        box = px.box(df, x="churn_pred", y="next_best_amount", labels={'churn_pred':'Прогноз відтоку','next_best_amount':'Сума (грн)'}, title="Порівняння рекомендованих сум: відтік vs не відтік", color_discrete_sequence=[OLIVE_LIGHT, OLIVE], template=TEMPLATE)
        rec = px.histogram(df, x='recency', nbins=20, labels={'recency':'Дні з останнього донату'}, title='Розподіл часу від останнього донату', color_discrete_sequence=[OLIVE], template=TEMPLATE)
        top10 = px.bar(df.nlargest(10,'next_best_amount'), x='user_id', y='next_best_amount', labels={'user_id':'ID користувача','next_best_amount':'Сума (грн)'}, title='Топ 10 користувачів за рекомендованою сумою', color_discrete_sequence=[OLIVE], template=TEMPLATE)
        data = df[["user_id","last_date","churn_proba","churn_pred","next_best_amount"]].to_dict('records')
        return roc, fi, hist, scat, box, rec, data, top10

    @app.callback(Output("download-data","data"), Input("download-table","n_clicks"), State("date-range","start_date"), State("date-range","end_date"), State("churn-proba-slider","value"), State("amount-threshold","value"), State("churn-pred-filter","value"), prevent_initial_call=True)
    def download_csv(n, start_date, end_date, thresh, amt_thresh, pred_filter):
        df = user_feats.copy()
        df = df[(df['last_date'] >= pd.to_datetime(start_date)) & (df['last_date'] <= pd.to_datetime(end_date))]
        df = df[df["churn_proba"]>=thresh]
        df = df[df["next_best_amount"]>=amt_thresh]
        if pred_filter in (0,1): df = df[df["churn_pred"]==pred_filter]
        return dcc.send_data_frame(df[["user_id","last_date","churn_proba","churn_pred","next_best_amount"]].to_csv, "churn_data.csv", index=False)
