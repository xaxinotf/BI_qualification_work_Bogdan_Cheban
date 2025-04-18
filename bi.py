# bi.py

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.cluster import KMeans

try:
    import dask.dataframe as dd
    USE_DASK = True
except ImportError:
    USE_DASK = False

try:
    from pyspark.sql import SparkSession
    USE_SPARK = True
    spark = SparkSession.builder.appName("BI_Application").getOrCreate()
except ImportError:
    USE_SPARK = False

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto

# імпорт модуля для Retention Cohorts
from retention_cohorts import get_cohort_controls, make_cohort_figure, make_cohort_size_figure

# ==================================================================
# Функція завантаження даних (підтримка Dask та PySpark)
# ==================================================================
DATA_DIR = "data"

def load_data(filename):
    path = os.path.join(DATA_DIR, filename)
    if USE_SPARK:
        return spark.read.option("header", True).csv(path).toPandas()
    elif USE_DASK:
        return dd.read_csv(path).compute()
    else:
        return pd.read_csv(path)

# ---------------------- Завантаження даних ----------------------
users_df = load_data("user.csv")
liqpay_orders_df = load_data("liqpay_order.csv")
requests_df = load_data("request.csv")
military_personnel_df = load_data("military_personnel.csv")
brigade_df = load_data("brigade.csv")

# ---------------------- Підготовка та обробка ----------------------
users_df["registration_date"] = pd.to_datetime(users_df["create_date"], errors="coerce")
users_df["duration_on_platform"] = (datetime.today() - users_df["registration_date"]).dt.days
liqpay_orders_df["create_date"] = pd.to_datetime(liqpay_orders_df["create_date"], errors="coerce")

# Злиття для донатної статистики
donations = liqpay_orders_df.merge(
    users_df[["id", "duration_on_platform", "user_role"]],
    left_on="user_id", right_on="id", how="left"
)
donation_stats = donations.groupby("user_id", observed=False).agg(
    total_donations=("amount", "sum"),
    transaction_count=("amount", "count")
).reset_index()
donation_stats["avg_donation_user"] = donation_stats["total_donations"] / donation_stats["transaction_count"]
donation_stats = donation_stats.merge(
    users_df[["id", "duration_on_platform", "user_role"]],
    left_on="user_id", right_on="id", how="left"
)
# Кореляційні розрахунки
corr_value = donation_stats["duration_on_platform"].corr(donation_stats["total_donations"])
numeric_cols = donation_stats.select_dtypes(include=[np.number]).columns
corr_matrix = donation_stats[numeric_cols].corr()

# ---------------------- KPI розрахунки ----------------------
total_donations = donation_stats["total_donations"].sum()
avg_donation = donation_stats["total_donations"].mean()
unique_donors = liqpay_orders_df["user_id"].nunique()
percent_donors = unique_donors / len(users_df) * 100
max_donation = donation_stats["total_donations"].max()
avg_duration_on_platform = donation_stats["duration_on_platform"].mean()

default_kpi_targets = {
    "Загальна сума донатів": 1_000_000,
    "Середній донат": 200,
    "Кількість унікальних донорів": 5000,
    "Відсоток користувачів, що донатять": 10,
    "Максимальний донат": 100_000,
    "Середня тривалість перебування на платформі (днів)": 365,
    "Кореляція (перебування vs донати)": 0.3
}

def evaluate_kpi(actual, target, higher_better=True):
    return "Досягнуто" if (actual >= target if higher_better else actual <= target) else "Не досягнуто"

def calculate_kpi_info(targets):
    return [
        {
            "name": "Загальна сума внесків",
            "actual": total_donations,
            "target": targets["Загальна сума донатів"],
            "status": evaluate_kpi(total_donations, targets["Загальна сума донатів"], True)
        },
        {
            "name": "Середній внесок",
            "actual": avg_donation,
            "target": targets["Середній донат"],
            "status": evaluate_kpi(avg_donation, targets["Середній донат"], True)
        },
        {
            "name": "Кількість унікальних донорів",
            "actual": unique_donors,
            "target": targets["Кількість унікальних донорів"],
            "status": evaluate_kpi(unique_donors, targets["Кількість унікальних донорів"], True)
        },
        {
            "name": "Відсоток користувачів, що вносять",
            "actual": percent_donors,
            "target": targets["Відсоток користувачів, що донатять"],
            "status": evaluate_kpi(percent_donors, targets["Відсоток користувачів, що донатять"], True)
        },
        {
            "name": "Максимальний внесок",
            "actual": max_donation,
            "target": targets["Максимальний донат"],
            "status": evaluate_kpi(max_donation, targets["Максимальний донат"], True)
        },
        {
            "name": "Середня тривалість перебування (днів)",
            "actual": avg_duration_on_platform,
            "target": targets["Середня тривалість перебування на платформі (днів)"],
            "status": evaluate_kpi(avg_duration_on_platform, targets["Середня тривалість перебування на платформі (днів)"], False)
        },
        {
            "name": "Кореляція (перебування vs внески)",
            "actual": corr_value,
            "target": targets["Кореляція (перебування vs донати)"],
            "status": evaluate_kpi(corr_value, targets["Кореляція (перебування vs донати)"], True)
        }
    ]

kpi_info = calculate_kpi_info(default_kpi_targets)

# ---------------------- Сегментація донорів (K‑Means) ----------------------
X = donation_stats[["duration_on_platform", "total_donations"]].fillna(0)
kmeans = KMeans(n_clusters=4, random_state=42)
donation_stats["cluster"] = kmeans.fit_predict(X)

# ---------------------- Конфігурація діаграм ----------------------
chart_config_default = {"height": 600, "width": 1200}
heatmap_config = {"height": 600, "width": 1200, "title": "Кореляційна матриця числових показників"}
hist_reg_config = {"height": 600, "width": 1200, "title": "Розподіл часу перебування (дні)"}
scatter_config = {
    "height": 600,
    "width": 1200,
    "title": "Середній внесок vs. Тривалість перебування",
    "labels": {"duration_on_platform": "Тривалість (днів)", "avg_donation_user": "Середній внесок"}
}
bar_config = {"height": 600, "width": 1200, "title": "Середній внесок по місяцях", "labels": {"month": "Місяць", "amount": "Середній внесок"}}
line_config = {"height": 600, "width": 1200, "title": "Сукупна сума внесків по днях", "labels": {"create_date": "Дата", "amount": "Сума внесків"}}

# побудова базових фігур
fig_heatmap = px.imshow(corr_matrix, text_auto=True, **heatmap_config)
fig_heatmap.update_xaxes(title_text="Показники")
fig_heatmap.update_yaxes(title_text="Показники")
fig_heatmap.add_annotation(
    text="Ця матриця демонструє силу та напрямок зв'язків між показниками.",
    xref="paper", yref="paper", x=0.5, y=-0.15, showarrow=False,
    font=dict(size=12, color="#666")
)

fig_hist_reg = px.histogram(users_df, x="duration_on_platform", nbins=50, **hist_reg_config)
fig_scatter = px.scatter(donation_stats, x="duration_on_platform", y="avg_donation_user", color="user_role", **scatter_config)

liqpay_orders_df["month"] = liqpay_orders_df["create_date"].dt.to_period("M").dt.to_timestamp()
monthly_avg = liqpay_orders_df.groupby("month")["amount"].mean().reset_index()
fig_avg_donation_month = px.bar(monthly_avg, x="month", y="amount", **bar_config)

orders_by_day = liqpay_orders_df.groupby(liqpay_orders_df["create_date"].dt.date)["amount"].sum().reset_index()
fig_line = px.line(orders_by_day, x="create_date", y="amount", **line_config)

# підготовка таблиці повороту (pivot) для Dash DataTable
donation_stats["duration_bin"] = pd.cut(donation_stats["duration_on_platform"], bins=10)
pivot_table = donation_stats.pivot_table(
    index="user_role", columns="duration_bin", values="total_donations",
    aggfunc="sum", fill_value=0
).reset_index()
pivot_table.columns = pivot_table.columns.map(str)
pivot_table_div = dash_table.DataTable(
    id='pivot-table',
    columns=[{"name": col, "id": col, "editable": True} for col in pivot_table.columns],
    data=pivot_table.to_dict("records"),
    filter_action="native", sort_action="native", page_size=15, export_format="csv",
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'center', 'padding': '5px', 'minWidth': '80px'},
    style_header={'backgroundColor': '#ddd', 'fontWeight': 'bold'},
    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f3f3f3"}]
)

# ---------------------- Аналіз запитів за бригадами ----------------------
military_personnel_df = military_personnel_df.rename(columns={"id": "mp_id"})
brigade_df = brigade_df.rename(columns={"id": "brigade_id"})
requests_merged = requests_df.merge(
    military_personnel_df, left_on="military_personnel_id", right_on="mp_id", how="left"
).merge(
    brigade_df, left_on="brigade_id", right_on="brigade_id", how="left"
)
grouped_requests = requests_merged.groupby("name").agg(
    total_requests=("id", "count"),
    total_amount=("amount", "sum"),
    average_amount=("amount", "mean")
).reset_index()
fig_requests_by_brigade = px.bar(
    grouped_requests, x="name", y="total_amount",
    title="Аналіз заявок: Загальна сума за бригадами",
    labels={"name": "Бригада", "total_amount": "Сума заявок"}
)
fig_requests_by_brigade.update_layout(**chart_config_default)

# ---------------------- Аналіз товарів ----------------------
if "product" not in requests_df.columns:
    def extract_product(desc):
        m = re.search(r"'([^']+)'", desc)
        return m.group(1) if m else "Невідомо"
    requests_df["product"] = requests_df["description"].apply(extract_product)

def get_product_aggregation(df):
    return df.groupby("product").agg(
        request_count=("id", "count"),
        total_amount=("amount", "sum"),
        average_amount=("amount", "mean")
    ).reset_index()

product_group = get_product_aggregation(requests_df)
fig_products = px.bar(
    product_group, x="product", y="request_count",
    title="Кількість заявок за товарами",
    labels={"product": "Товар", "request_count": "Кількість заявок"}
)
fig_products.update_layout(**chart_config_default)
product_table = dash_table.DataTable(
    id="product-table",
    columns=[{"name": col, "id": col, "editable": True} for col in product_group.columns],
    data=product_group.to_dict("records"),
    filter_action="native", sort_action="native", page_size=15, export_format="csv",
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'center', 'padding': '5px', 'minWidth': '80px'},
    style_header={'backgroundColor': '#ddd', 'fontWeight': 'bold'},
    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f3f3f3"}]
)

# ---------------------- Зіркова схема OLAP‑куба ----------------------
def create_star_schema_cytoscape():
    fact = "Liqpay_order"
    dimensions = [
        "User","Volunteer","Military Personnel","Request","Levy","Volunteer_Levy",
        "Report","Attachment","Brigade Codes","Add Request","Email Template",
        "Email Notification","Email Recipient","Email Attachment","AI Chat Messages"
    ]
    elements = [{
        'data': {'id': 'fact', 'label': fact},
        'position': {'x': 600, 'y': 400},
        'classes': 'fact'
    }]
    n = len(dimensions)
    radius = 400
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    for i, dim in enumerate(dimensions):
        x = 600 + radius * np.cos(angles[i])
        y = 400 + radius * np.sin(angles[i])
        elements.append({
            'data': {'id': f'dim{i}', 'label': dim},
            'position': {'x': x, 'y': y},
            'classes': 'dimension'
        })
        elements.append({'data': {'source': 'fact', 'target': f'dim{i}'}})
    return elements

cyto_elements = create_star_schema_cytoscape()

# ---------------------- KPI Cards тa Controls ----------------------
def create_kpi_div(kpi_info):
    kpi_divs = []
    for kpi in kpi_info:
        value_str = f"{kpi['actual']:.2f}" if isinstance(kpi["actual"], (int, float)) else str(kpi["actual"])
        target_str = f"{kpi['target']:.2f}" if isinstance(kpi["target"], (int, float)) else str(kpi["target"])
        status_class = "success" if kpi["status"] == "Досягнуто" else "failure"
        kpi_divs.append(html.Div([
            html.H3(kpi["name"], className="kpi-heading"),
            html.P(f"Факт: {value_str}", className="kpi-fact"),
            html.P(f"Ціль: {target_str}", className="kpi-target"),
            html.P(kpi["status"], className=f"kpi-status {status_class}")
        ], className="kpi-card"))
    return html.Div(kpi_divs, className="kpi-container")

kpi_div = create_kpi_div(kpi_info)

kpi_controls = html.Div([
    html.H2("Налаштування KPI"),
    html.Div([
        html.Div([
            html.Label("Загальна сума внесків (ціль):", className="input-label"),
            dcc.Input(id="target-total-donations", type="number",
                      value=default_kpi_targets["Загальна сума донатів"], className="input-field")
        ], className="kpi-config-group"),
        html.Div([
            html.Label("Середній внесок (ціль):", className="input-label"),
            dcc.Input(id="target-avg-donation", type="number",
                      value=default_kpi_targets["Середній донат"], className="input-field")
        ], className="kpi-config-group"),
        html.Div([
            html.Label("Кількість унікальних донорів (ціль):", className="input-label"),
            dcc.Input(id="target-unique-donors", type="number",
                      value=default_kpi_targets["Кількість унікальних донорів"], className="input-field")
        ], className="kpi-config-group"),
        html.Div([
            html.Label("Відсоток користувачів, що вносять (ціль):", className="input-label"),
            dcc.Input(id="target-percent-donors", type="number",
                      value=default_kpi_targets["Відсоток користувачів, що донатять"], className="input-field")
        ], className="kpi-config-group"),
        html.Div([
            html.Label("Максимальний внесок (ціль):", className="input-label"),
            dcc.Input(id="target-max-donations", type="number",
                      value=default_kpi_targets["Максимальний донат"], className="input-field")
        ], className="kpi-config-group"),
        html.Div([
            html.Label("Середня тривалість перебування (днів):", className="input-label"),
            dcc.Input(id="target-avg-reg-duration", type="number",
                      value=default_kpi_targets["Середня тривалість перебування на платформі (днів)"], className="input-field")
        ], className="kpi-config-group"),
        html.Div([
            html.Label("Кореляція (ціль):", className="input-label"),
            dcc.Input(id="target-corr", type="number", step=0.01,
                      value=default_kpi_targets["Кореляція (перебування vs донати)"], className="input-field")
        ], className="kpi-config-group"),
        html.Button("Оновити KPI", id="update-kpi-btn", n_clicks=0, className="kpi-config-btn")
    ], className="kpi-config-panel"),
    html.Div(id="kpi-notifications", className="kpi-notification")
], className="kpi-controls")

# ---------------------- Побудова Dash-додатку ----------------------
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, "https://fonts.googleapis.com/css?family=Montserrat:400,600&display=swap"]
)

app.layout = html.Div(className="container-fluid", children=[
    html.Div(className="row", children=[
        html.Div(className="col-md-3 sidebar", children=[
            html.H2("Меню", className="text-center text-success"),
            html.Hr(),
            dcc.Tabs(
                id="sidebar-tabs",
                value="tab-analytics",
                vertical=True,
                children=[
                    dcc.Tab(label="Аналітика", value="tab-analytics"),
                    dcc.Tab(label="Зіркова схема", value="tab-star"),
                    dcc.Tab(label="Сегментація донорів", value="tab-segmentation"),
                    dcc.Tab(label="Прогнозування", value="tab-forecast"),
                    dcc.Tab(label="Керування KPI", value="tab-kpi"),
                    dcc.Tab(label="Запити", value="tab-requests"),
                    dcc.Tab(label="Аналіз товарів", value="tab-products"),
                    dcc.Tab(label="Волонтерський аналіз", value="tab-volunteers"),
                    dcc.Tab(label="Аналіз зборів", value="tab-levy"),
                    dcc.Tab(label="Retention Cohorts", value="tab-cohorts"),
                ]
            )
        ]),
        html.Div(className="col-md-9 main", id="main-content")
    ])
])

# ---------------------- Callback перемикання вмісту ----------------------
@app.callback(
    Output("main-content", "children"),
    Input("sidebar-tabs", "value")
)
def render_tab_content(tab_value):
    if tab_value == "tab-analytics":
        return html.Div([
            html.Div(className="filter-panel", children=[
                html.Label("Фільтр за роллю користувача:"),
                dcc.Dropdown(
                    id="role-filter",
                    options=[{"label": r, "value": r} for r in sorted(users_df["user_role"].unique())],
                    value=sorted(users_df["user_role"].unique()),
                    multi=True
                ),
                html.Label("Діапазон часу перебування (днів):"),
                dcc.RangeSlider(
                    id="reg-duration-slider",
                    min=int(users_df["duration_on_platform"].min()),
                    max=int(users_df["duration_on_platform"].max()),
                    value=[int(users_df["duration_on_platform"].min()), int(users_df["duration_on_platform"].max())],
                    marks={i: str(i) for i in np.linspace(users_df["duration_on_platform"].min(),
                                                          users_df["duration_on_platform"].max(), 10)}
                )
            ]),
            html.Div([html.H2("Середній внесок vs. Тривалість перебування"),
                      dcc.Graph(id="scatter-graph", figure=fig_scatter)]),
            html.Div([html.H2("Середній внесок по місяцях"),
                      dcc.Graph(id="hist-donations", figure=fig_avg_donation_month)]),
            html.Div([html.H2("Сукупна сума внесків по днях"),
                      dcc.Graph(id="line-orders", figure=fig_line)]),
            html.Div([html.H2("Кореляційна матриця"),
                      dcc.Graph(id="heatmap", figure=fig_heatmap)]),
            html.Div([html.H2("Розподіл часу перебування (дні)"),
                      dcc.Graph(id="hist-reg", figure=fig_hist_reg)]),
            html.Div(id="kpi-div-updated", children=[html.H2("Ключові показники (KPI)"), kpi_div])
        ])
    elif tab_value == "tab-star":
        return html.Div([
            html.H2("Зіркова схема OLAP‑куба"),
            cyto.Cytoscape(
                id='star-schema',
                elements=cyto_elements,
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '700px'},
                stylesheet=[
                    {'selector': 'node',
                     'style': {'label': 'data(label)', 'text-valign': 'center',
                               'color': 'var(--color-text)', 'background-color': 'var(--color-primary)',
                               'width': 50, 'height': 50, 'font-size': 12,
                               'border-width': 2, 'border-color': '#ddd'}},
                    {'selector': 'node.fact',
                     'style': {'background-color': 'var(--color-secondary)', 'width': 70, 'height': 70, 'font-size': 14}},
                    {'selector': 'edge',
                     'style': {'line-color': '#ccc', 'width': 2}}
                ]
            ),
            html.Div(id="drill-down", className="mt-4")
        ], className="tab-content")
    elif tab_value == "tab-segmentation":
        return html.Div([
            html.H2("Кластеризація донорів (K-Means)"),
            dcc.Dropdown(
                id="cluster-filter",
                options=[{"label": f"Кластер {i}", "value": i} for i in sorted(donation_stats["cluster"].unique())],
                value=sorted(donation_stats["cluster"].unique()),
                multi=True
            ),
            html.Div([dcc.Graph(id="cluster-scatter"), dcc.Graph(id="cluster-box")])
        ], className="tab-content")
    elif tab_value == "tab-forecast":
        return html.Div([
            html.H2("Прогнозування внесків"),
            dcc.Slider(
                id="forecast-horizon-slider", min=30, max=180, step=10, value=90,
                marks={i: str(i) for i in range(30, 181, 10)}
            ),
            html.Div(id="forecast-output", className="mt-4")
        ], className="tab-content")
    elif tab_value == "tab-kpi":
        return html.Div([
            kpi_controls,
            html.Div(id="kpi-div-updated",
                     children=[html.H2("Ключові показники (KPI)"), kpi_div]),
            html.Div([html.H2("Історія KPI"), dcc.Graph(id="kpi-history", figure=px.line(
                pd.DataFrame({
                    "Дата": sorted([datetime.today() - timedelta(days=i) for i in range(30)]),
                    "Сукупна сума внесків": np.cumsum(np.random.randint(1000, 5000, size=30))
                }),
                x="Дата", y="Сукупна сума внесків",
                labels={"Сукупна сума внесків": "Сума внесків", "Дата": "Дата"}
            ).update_layout(**chart_config_default))])
        ], className="tab-content")
    elif tab_value == "tab-requests":
        return html.Div([
            html.H2("Аналіз заявок за бригадами"),
            dcc.Graph(id="requests-analysis-graph", figure=fig_requests_by_brigade),
            dcc.Dropdown(
                id="request-metric-dropdown",
                options=[
                    {"label": "Кількість заявок", "value": "total_requests"},
                    {"label": "Сума заявок", "value": "total_amount"},
                    {"label": "Середня сума заявки", "value": "average_amount"}
                ],
                value="total_amount", clearable=False
            )
        ], className="tab-content")
    elif tab_value == "tab-products":
        return html.Div([
            html.H2("Аналіз заявок за товарами"),
            dcc.Input(id="product-topic-filter", type="text",
                      placeholder="Введіть тему запиту...", className="form-control mb-2"),
            dcc.Graph(id="product-bar-chart", figure=fig_products),
            dcc.Dropdown(
                id="product-metric-dropdown",
                options=[
                    {"label": "Кількість заявок", "value": "request_count"},
                    {"label": "Загальна сума заявок", "value": "total_amount"},
                    {"label": "Середня сума заявки", "value": "average_amount"}
                ],
                value="request_count", clearable=False, className="w-50 mb-4"
            ),
            html.H2("Таблиця аналізу заявок за товарами"),
            product_table
        ], className="tab-content")
    elif tab_value == "tab-volunteers":
        from analysis_of_volunteer import VolunteerAnalysis
        va = VolunteerAnalysis(data_dir="data")
        fig_vol_scatter, fig_vol_topwords = va.create_plots(
            cluster_chart_config={
                "width": 1300, "height": 700,
                "title": "Кластеризація волонтерів: Репутація vs. Тональність",
                "legend": {"title": {"text": "Кластер"}}
            },
            topwords_chart_config={
                "width": 1300, "height": 700,
                "title": "Тематичний аналіз: Топ-10 ключових слів"
            }
        )
        return html.Div([
            html.H2("Аналіз волонтерської діяльності та репутації"),
            html.H3("Кластеризація волонтерів"),
            dcc.Graph(id="volunteer-scatter", figure=fig_vol_scatter),
            html.H3("Тематичний аналіз відгуків"),
            dcc.Graph(id="volunteer-topwords", figure=fig_vol_topwords)
        ], className="tab-content")
    elif tab_value == "tab-levy":
        from analysis_of_fees_and_reporting import LevyAnalysis
        la = LevyAnalysis(data_dir="data")
        fig_levy_scatter, fig_levy_trend, fig_levy_corr = la.create_plots()
        return html.Div([
            html.H2("Аналіз зборів та звітності"),
            dcc.Graph(id="levy-scatter", figure=fig_levy_scatter),
            dcc.Graph(id="levy-trend", figure=fig_levy_trend),
            dcc.Graph(id="levy-corr", figure=fig_levy_corr)
        ], className="tab-content")
    elif tab_value == "tab-cohorts":
        return html.Div([
            html.H2("Утримання донорів (Retention Cohorts)"),
            get_cohort_controls(),
            dcc.Graph(id="cohort-heatmap", figure=make_cohort_figure(liqpay_orders_df, 12, 5)),
            dcc.Graph(id="cohort-sizes", figure=make_cohort_size_figure(liqpay_orders_df, 5))
        ], className="tab-content")
    else:
        return html.Div("Виберіть пункт меню.", className="tab-content")

# ---------------------- Основні Callback-и ----------------------

# Оновлення основної аналітики
@app.callback(
    [Output("scatter-graph", "figure"),
     Output("hist-donations", "figure"),
     Output("hist-reg", "figure")],
    [Input("role-filter", "value"),
     Input("reg-duration-slider", "value")]
)
def update_main_analytics(selected_roles, reg_duration_range):
    filtered = donation_stats[
        (donation_stats["user_role"].isin(selected_roles)) &
        (donation_stats["duration_on_platform"] >= reg_duration_range[0]) &
        (donation_stats["duration_on_platform"] <= reg_duration_range[1])
    ]
    fig1 = px.scatter(
        filtered, x="duration_on_platform", y="avg_donation_user",
        color="user_role",
        title="Середній внесок vs. Тривалість перебування",
        labels={"duration_on_platform": "Тривалість (днів)", "avg_donation_user": "Середній внесок"}
    )
    fig1.update_layout(**chart_config_default)

    liqpay_orders_df["month"] = liqpay_orders_df["create_date"].dt.to_period("M").dt.to_timestamp()
    monthly_avg2 = liqpay_orders_df.groupby("month")["amount"].mean().reset_index()
    fig2 = px.bar(
        monthly_avg2, x="month", y="amount",
        title="Середній внесок по місяцях",
        labels={"month": "Місяць", "amount": "Середній внесок"}
    )
    fig2.update_layout(**chart_config_default)

    df3 = users_df[
        (users_df["duration_on_platform"] >= reg_duration_range[0]) &
        (users_df["duration_on_platform"] <= reg_duration_range[1])
    ]
    fig3 = px.histogram(df3, x="duration_on_platform", nbins=50,
                        title="Розподіл часу перебування (дні)",
                        labels={"duration_on_platform": "Тривалість (днів)"})
    fig3.update_layout(**chart_config_default)

    return fig1, fig2, fig3

# Дрилл-даун для Star Schema
@app.callback(
    Output("drill-down", "children"),
    Input("star-schema", "tapNodeData")
)
def drill_down(nodeData):
    if not nodeData:
        return "Натисніть на вузол для перегляду деталей."
    label = nodeData.get("label")
    if label == "User":
        fig = px.histogram(users_df, x="duration_on_platform", nbins=20,
                           title="Деталізація: Тривалість перебування")
        return dcc.Graph(figure=fig)
    if label == "Request":
        try:
            df = load_data("request.csv")
            fig = px.histogram(df, x="amount", nbins=30, title="Деталізація: Суми заявок")
            return dcc.Graph(figure=fig)
        except Exception as e:
            return f"Помилка: {e}"
    if label == "Liqpay_order":
        ob = liqpay_orders_df.groupby(liqpay_orders_df["create_date"].dt.date)["amount"].sum().reset_index()
        fig = px.line(ob, x="create_date", y="amount", title="Деталізація: Тренд транзакцій")
        return dcc.Graph(figure=fig)
    return f"Вибрано: {label}. Деталі недоступні."

# Кластери донорів
@app.callback(
    [Output("cluster-scatter", "figure"), Output("cluster-box", "figure")],
    Input("cluster-filter", "value")
)
def update_clusters(selected_clusters):
    filt = donation_stats[donation_stats["cluster"].isin(selected_clusters)]
    fig_sc = px.scatter(
        filt, x="duration_on_platform", y="total_donations",
        color="cluster",
        title="Кластеризація: Тривалість vs. Сума внесків",
        labels={"duration_on_platform": "Тривалість (днів)", "total_donations": "Сума внесків", "cluster": "Кластер"}
    )
    fig_sc.update_layout(**chart_config_default)
    fig_box = px.box(
        filt, x="cluster", y="total_donations",
        title="Розподіл внесків за кластерами",
        labels={"cluster": "Кластер", "total_donations": "Сума внесків"}
    )
    fig_box.update_layout(**chart_config_default)
    return fig_sc, fig_box

# Прогнозування
@app.callback(
    Output("forecast-output", "children"),
    Input("forecast-horizon-slider", "value")
)
def update_forecast(horizon):
    try:
        fig = forecast_donations(horizon)
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.P(f"Помилка прогнозування: {e}", style={"color": "red"})

# Оновлення KPI
@app.callback(
    [Output("kpi-notifications", "children"), Output("kpi-div-updated", "children")],
    Input("update-kpi-btn", "n_clicks"),
    State("target-total-donations", "value"), State("target-avg-donation", "value"),
    State("target-unique-donors", "value"), State("target-percent-donors", "value"),
    State("target-max-donations", "value"), State("target-avg-reg-duration", "value"),
    State("target-corr", "value"), State("kpi-div-updated", "children")
)
def update_kpi(n, total_t, avg_t, uniq_t, perc_t, max_t, avg_reg_t, corr_t, old):
    if n == 0:
        return "", old
    new_targets = {
        "Загальна сума донатів": total_t,
        "Середній донат": avg_t,
        "Кількість унікальних донорів": uniq_t,
        "Відсоток користувачів, що донатять": perc_t,
        "Максимальний донат": max_t,
        "Середня тривалість перебування на платформі (днів)": avg_reg_t,
        "Кореляція (перебування vs донати)": corr_t
    }
    new_info = calculate_kpi_info(new_targets)
    msgs = []
    for k in new_info:
        if k["status"] == "Не досягнуто":
            msgs.append(html.P(
                f"Попередження! {k['name']} не досягнуто: {k['actual']:.2f} vs {k['target']:.2f}.",
                style={"color": "red", "textAlign": "center"}
            ))
    if not msgs:
        msgs = html.P("Всі KPI відповідають встановленим цілям.", style={"color": "green", "textAlign": "center"})
    updated_div = create_kpi_div(new_info)
    content = html.Div([
        html.H2("Ключові показники (KPI)", className="section-heading"),
        updated_div
    ], className="kpi-section")
    return msgs, content

# Аналіз заявок
@app.callback(
    Output("requests-analysis-graph", "figure"),
    Input("request-metric-dropdown", "value")
)
def update_requests_analysis(metric):
    if grouped_requests.empty:
        return go.Figure()
    y_label = {
        "total_requests": "Кількість заявок",
        "total_amount": "Сума заявок",
        "average_amount": "Середня сума заявки"
    }[metric]
    fig = px.bar(
        grouped_requests, x="name", y=metric,
        title=f"Аналіз заявок: {y_label} за бригадами",
        labels={"name": "Бригада", metric: y_label}
    )
    fig.update_layout(**chart_config_default)
    return fig

# Аналіз товарів
@app.callback(
    Output("product-bar-chart", "figure"),
    [Input("product-metric-dropdown", "value"), Input("product-topic-filter", "value")]
)
def update_product_chart(metric, topic):
    df = requests_df.copy()
    if topic and topic.strip():
        df = df[df["description"].str.contains(topic, case=False, na=False)]
    pg = get_product_aggregation(df)
    y_label = {
        "request_count": "Кількість заявок",
        "total_amount": "Загальна сума заявок",
        "average_amount": "Середня сума заявки"
    }[metric]
    fig = px.bar(
        pg, x="product", y=metric,
        title=f"Показник: {y_label} за товарами",
        labels={"product": "Товар", metric: y_label}
    )
    fig.update_layout(**chart_config_default)
    return fig

# ---------------------- Callback для Retention Cohorts ----------------------
@app.callback(
    [Output("cohort-heatmap", "figure"),
     Output("cohort-sizes",   "figure")],
    [Input("cohort-max-periods", "value"),
     Input("cohort-min-users",   "value")]
)
def update_cohort_charts(max_months, min_users):
    return (
        make_cohort_figure(liqpay_orders_df, max_months, min_users),
        make_cohort_size_figure(liqpay_orders_df,     min_users)
    )
# ---------------------- Запуск ----------------------
if __name__ == "__main__":
    app.run(debug=True)
