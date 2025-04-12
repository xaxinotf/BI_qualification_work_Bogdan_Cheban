import os, re, numpy as np, pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.cluster import KMeans

# Спроба використання Dask для роботи з великими наборами даних (якщо встановлено)
try:
    import dask.dataframe as dd
    USE_DASK = True
except ImportError:
    USE_DASK = False

# Використання Dash Bootstrap Components для сучасного оформлення
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto

# ---------------------- Налаштування директорії та завантаження даних ---------------------- #
DATA_DIR = "data"

def load_data(filename):
    path = os.path.join(DATA_DIR, filename)
    if USE_DASK:
        return dd.read_csv(path).compute()
    else:
        return pd.read_csv(path)

# ---------------------- Завантаження основних даних ---------------------- #
users_df = load_data("user.csv")
liqpay_orders_df = load_data("liqpay_order.csv")
requests_df = load_data("request.csv")

# ---------------------- Обробка даних ---------------------- #
users_df["registration_date"] = pd.to_datetime(users_df["create_date"], errors="coerce")
users_df["duration_on_platform"] = (datetime.today() - users_df["registration_date"]).dt.days
liqpay_orders_df["create_date"] = pd.to_datetime(liqpay_orders_df["create_date"], errors="coerce")

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
corr_value = donation_stats["duration_on_platform"].corr(donation_stats["total_donations"])
numeric_cols = donation_stats.select_dtypes(include=[np.number]).columns
corr_matrix = donation_stats[numeric_cols].corr()

# ---------------------- KPI Розрахунки ---------------------- #
total_donations = donation_stats["total_donations"].sum()
avg_donation = donation_stats["total_donations"].mean()
unique_donors = liqpay_orders_df["user_id"].nunique()
percent_donors = (unique_donors / len(users_df)) * 100
max_donation = donation_stats["total_donations"].max()
avg_duration_on_platform = donation_stats["duration_on_platform"].mean()

default_kpi_targets = {
    "Загальна сума донатів": 1_000_000,
    "Середній донат": 200,
    "Кількість унікальних донорів": 5000,
    "Відсоток користувачів, що донатять": 10,
    "Максимальний донат": 100000,
    "Середня тривалість перебування на платформі (днів)": 365,
    "Кореляція (перебування vs донати)": 0.3
}

def evaluate_kpi(actual, target, higher_better=True):
    return "Досягнуто" if (actual >= target if higher_better else actual <= target) else "Не досягнуто"

def calculate_kpi_info(targets):
    info = [
        {"name": "Загальна сума внесків", "actual": total_donations, "target": targets["Загальна сума донатів"],
         "status": evaluate_kpi(total_donations, targets["Загальна сума донатів"], higher_better=True)},
        {"name": "Середній внесок", "actual": avg_donation, "target": targets["Середній донат"],
         "status": evaluate_kpi(avg_donation, targets["Середній донат"], higher_better=True)},
        {"name": "Кількість унікальних донорів", "actual": unique_donors,
         "target": targets["Кількість унікальних донорів"],
         "status": evaluate_kpi(unique_donors, targets["Кількість унікальних донорів"], higher_better=True)},
        {"name": "Відсоток користувачів, що вносять", "actual": percent_donors,
         "target": targets["Відсоток користувачів, що донатять"],
         "status": evaluate_kpi(percent_donors, targets["Відсоток користувачів, що донатять"], higher_better=True)},
        {"name": "Максимальний внесок", "actual": max_donation, "target": targets["Максимальний донат"],
         "status": evaluate_kpi(max_donation, targets["Максимальний донат"], higher_better=True)},
        {"name": "Середня тривалість перебування (днів)", "actual": avg_duration_on_platform,
         "target": targets["Середня тривалість перебування на платформі (днів)"],
         "status": evaluate_kpi(avg_duration_on_platform, targets["Середня тривалість перебування на платформі (днів)"],
                                higher_better=False)},
        {"name": "Кореляція (перебування vs внески)", "actual": corr_value,
         "target": targets["Кореляція (перебування vs донати)"],
         "status": evaluate_kpi(corr_value, targets["Кореляція (перебування vs донати)"], higher_better=True)}
    ]
    return info

kpi_info = calculate_kpi_info(default_kpi_targets)

# ---------------------- Сегментація донорів (K‑Means) ---------------------- #
X = donation_stats[["duration_on_platform", "total_donations"]].fillna(0)
kmeans = KMeans(n_clusters=4, random_state=42)
donation_stats["cluster"] = kmeans.fit_predict(X)

# ---------------------- Конфігуровувальні налаштування діаграм ---------------------- #
chart_config_default = {"height": 600, "width": 1200}
heatmap_config = {"height": 600, "width": 1200, "title": "Кореляційна матриця числових показників"}
hist_reg_config = {"height": 600, "width": 1200, "title": "Розподіл часу перебування на платформі (дні)"}
scatter_config = {"height": 600, "width": 1200, "title": "Середній внесок vs. Тривалість перебування",
                  "labels": {"duration_on_platform": "Тривалість (дні)", "avg_donation_user": "Середній внесок"}}
bar_config = {"height": 600, "width": 1200, "title": "Середній внесок по місяцях",
              "labels": {"month": "Місяць", "amount": "Середній внесок"}}
line_config = {"height": 600, "width": 1200, "title": "Сукупна сума внесків по днях",
               "labels": {"create_date": "Дата", "amount": "Сума внесків"}}

# ---------------------- Побудова діаграм ---------------------- #
fig_heatmap = px.imshow(corr_matrix, text_auto=True, **heatmap_config)
fig_heatmap.update_xaxes(title_text="Показники")
fig_heatmap.update_yaxes(title_text="Показники")
fig_heatmap.add_annotation(
    text="Ця матриця демонструє силу та напрямок зв'язків між показниками.",
    xref="paper", yref="paper", x=0.5, y=-0.15, showarrow=False,
    font=dict(size=12, color="#666")
)

fig_hist_reg = px.histogram(users_df, x="duration_on_platform", nbins=50, **hist_reg_config)
fig_scatter = px.scatter(donation_stats, x="duration_on_platform", y="avg_donation_user",
                         color="user_role", **scatter_config)
liqpay_orders_df["month"] = liqpay_orders_df["create_date"].dt.to_period("M").dt.to_timestamp()
monthly_avg = liqpay_orders_df.groupby("month")["amount"].mean().reset_index()
fig_avg_donation_month = px.bar(monthly_avg, x="month", y="amount", **bar_config)
orders_by_day = liqpay_orders_df.groupby(liqpay_orders_df["create_date"].dt.date)["amount"].sum().reset_index()
fig_line = px.line(orders_by_day, x="create_date", y="amount", **line_config)

donation_stats["duration_bin"] = pd.cut(donation_stats["duration_on_platform"], bins=10)
pivot_table = donation_stats.pivot_table(index="user_role", columns="duration_bin", values="total_donations",
                                           aggfunc="sum", fill_value=0).reset_index()
pivot_table.columns = pivot_table.columns.map(str)
pivot_table_div = dash_table.DataTable(
    id='pivot-table',
    columns=[{"name": str(col), "id": str(col), "editable": True} for col in pivot_table.columns],
    data=pivot_table.to_dict('records'),
    filter_action="native",
    sort_action="native",
    page_size=15,
    export_format="csv",
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'center', 'padding': '5px', 'minWidth': '80px'},
    style_header={'backgroundColor': '#ddd', 'fontWeight': 'bold'},
    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f3f3f3"}]
)

# ---------------------- Аналіз запитів за бригадами ---------------------- #
military_personnel_df = load_data("military_personnel.csv")
brigade_df = load_data("brigade.csv")
military_personnel_df = military_personnel_df.rename(columns={"id": "mp_id"})
brigade_df = brigade_df.rename(columns={"id": "brigade_id"})
requests_merged = requests_df.merge(military_personnel_df, left_on="military_personnel_id", right_on="mp_id", how="left")
requests_merged = requests_merged.merge(brigade_df, left_on="brigade_id", right_on="brigade_id", how="left")
grouped_requests = requests_merged.groupby("name").agg(
    total_requests=("id", "count"),
    total_amount=("amount", "sum"),
    average_amount=("amount", "mean")
).reset_index()
fig_requests_by_brigade = px.bar(grouped_requests, x="name", y="total_amount",
                                 title="Аналіз заявок: Загальна сума за бригадами",
                                 labels={"name": "Бригада", "total_amount": "Сума заявок"})
fig_requests_by_brigade.update_layout(**chart_config_default)

# ---------------------- Аналіз товарів (з додатковою фільтрацією теми) ---------------------- #
if "product" not in requests_df.columns:
    def extract_product(desc):
        m = re.search(r"'([^']+)'", desc)
        return m.group(1) if m else "Невідомо"
    requests_df["product"] = requests_df["description"].apply(extract_product)

def get_product_aggregation(df):
    product_group = df.groupby("product").agg(
        request_count=("id", "count"),
        total_amount=("amount", "sum"),
        average_amount=("amount", "mean")
    ).reset_index()
    return product_group

product_group = get_product_aggregation(requests_df)
fig_products = px.bar(product_group, x="product", y="request_count",
                      title="Кількість заявок за товарами",
                      labels={"product": "Товар", "request_count": "Кількість заявок"})
fig_products.update_layout(**chart_config_default)
product_table = dash_table.DataTable(
    id="product-table",
    columns=[{"name": col, "id": col, "editable": True} for col in product_group.columns],
    data=product_group.to_dict("records"),
    filter_action="native",
    sort_action="native",
    page_size=15,
    export_format="csv",
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'center', 'padding': '5px', 'minWidth': '80px'},
    style_header={'backgroundColor': '#ddd', 'fontWeight': 'bold'},
    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f3f3f3"}]
)

# ---------------------- Зіркова схема OLAP‑куба ---------------------- #
def create_star_schema_cytoscape():
    fact = "Liqpay_order"
    dimensions = [
        "User", "Volunteer", "Military Personnel", "Request", "Levy",
        "Volunteer_Levy", "Report", "Attachment", "Brigade Codes", "Add Request",
        "Email Template", "Email Notification", "Email Recipient", "Email Attachment", "AI Chat Messages"
    ]
    elements = []
    elements.append({
        'data': {'id': 'fact', 'label': fact},
        'position': {'x': 600, 'y': 400},
        'classes': 'fact'
    })
    n = len(dimensions)
    radius = 400
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i, dim in enumerate(dimensions):
        x = 600 + radius * np.cos(angles[i])
        y = 400 + radius * np.sin(angles[i])
        elements.append({
            'data': {'id': f'dim{i}', 'label': dim},
            'position': {'x': x, 'y': y},
            'classes': 'dimension'
        })
        elements.append({
            'data': {'source': 'fact', 'target': f'dim{i}'}
        })
    return elements

cyto_elements = create_star_schema_cytoscape()

# ---------------------- KPI Cards ---------------------- #
def create_kpi_div(kpi_info):
    kpi_divs = []
    for kpi in kpi_info:
        value_str = f"{kpi['actual']:.2f}" if isinstance(kpi["actual"], (float, int)) else f"{kpi['actual']}"
        target_str = f"{kpi['target']:.2f}" if isinstance(kpi["target"], (float, int)) else f"{kpi['target']}"
        kpi_divs.append(
            html.Div([
                html.H3(kpi["name"], style={"color": "#333", "textAlign": "center"}),
                html.P(f"Факт: {value_str}", style={"color": "#555", "textAlign": "center"}),
                html.P(f"Ціль: {target_str}", style={"color": "#4CAF50", "textAlign": "center"}),
                html.P(f"Статус: {kpi['status']}", style={"color": "#666", "textAlign": "center"})
            ], style={
                "border": "2px solid #ddd", "padding": "20px", "margin": "10px",
                "backgroundColor": "#fff", "width": "calc(33% - 20px)", "display": "inline-block",
                "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.05)"
            })
        )
    return html.Div(kpi_divs, style={"display": "flex", "flexWrap": "wrap", "justifyContent": "space-around"})

kpi_div = create_kpi_div(kpi_info)

ts_df = liqpay_orders_df.groupby(liqpay_orders_df["create_date"].dt.date)["amount"].sum().reset_index()
ts_df.columns = ["ds", "y"]
ts_df["ds"] = pd.to_datetime(ts_df["ds"])

def forecast_donations(horizon_days):
    model = Prophet(daily_seasonality=True)
    model.fit(ts_df)
    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)
    fig_forecast = plot_plotly(model, forecast)
    fig_forecast.update_layout(title=f"Прогноз внесків на наступні {horizon_days} днів", **chart_config_default)
    return fig_forecast

kpi_controls = html.Div([
    html.H2("Налаштування KPI", style={"color": "#333", "textAlign": "center"}),
    html.Div([
        html.Label("Загальна сума внесків (ціль):", style={"color": "#333"}),
        dcc.Input(id="target-total-donations", type="number",
                  value=default_kpi_targets["Загальна сума донатів"],
                  style={"border": "1px solid #ddd", "padding": "5px", "width": "100%"})
    ], style={"margin": "10px"}),
    html.Div([
        html.Label("Середній внесок (ціль):", style={"color": "#333"}),
        dcc.Input(id="target-avg-donation", type="number",
                  value=default_kpi_targets["Середній донат"],
                  style={"border": "1px solid #ddd", "padding": "5px", "width": "100%"})
    ], style={"margin": "10px"}),
    html.Div([
        html.Label("Кількість унікальних донорів (ціль):", style={"color": "#333"}),
        dcc.Input(id="target-unique-donors", type="number",
                  value=default_kpi_targets["Кількість унікальних донорів"],
                  style={"border": "1px solid #ddd", "padding": "5px", "width": "100%"})
    ], style={"margin": "10px"}),
    html.Div([
        html.Label("Відсоток користувачів, що вносять (ціль):", style={"color": "#333"}),
        dcc.Input(id="target-percent-donors", type="number",
                  value=default_kpi_targets["Відсоток користувачів, що донатять"],
                  style={"border": "1px solid #ddd", "padding": "5px", "width": "100%"})
    ], style={"margin": "10px"}),
    html.Div([
        html.Label("Максимальний внесок (ціль):", style={"color": "#333"}),
        dcc.Input(id="target-max-donations", type="number",
                  value=default_kpi_targets["Максимальний донат"],
                  style={"border": "1px solid #ddd", "padding": "5px", "width": "100%"})
    ], style={"margin": "10px"}),
    html.Div([
        html.Label("Середня тривалість перебування на платформі (днів):", style={"color": "#333"}),
        dcc.Input(id="target-avg-reg-duration", type="number",
                  value=default_kpi_targets["Середня тривалість перебування на платформі (днів)"],
                  style={"border": "1px solid #ddd", "padding": "5px", "width": "100%"})
    ], style={"margin": "10px"}),
    html.Div([
        html.Label("Кореляція (ціль):", style={"color": "#333"}),
        dcc.Input(id="target-corr", type="number",
                  value=default_kpi_targets["Кореляція (перебування vs донати)"],
                  step=0.01,
                  style={"border": "1px solid #ddd", "padding": "5px", "width": "100%"})
    ], style={"margin": "10px"}),
    html.Button("Оновити KPI", id="update-kpi-btn", n_clicks=0,
                style={"backgroundColor": "#4CAF50", "color": "#fff", "padding": "10px 20px",
                       "border": "none", "cursor": "pointer", "margin": "10px", "width": "100%"}),
    html.Div(id="kpi-notifications", style={"marginTop": "20px", "color": "#333", "textAlign": "center"})
], style={"border": "2px solid #ddd", "padding": "20px", "margin": "20px", "backgroundColor": "#fff"})

# ---------------------- Додаткове оформлення через Bootstrap ---------------------- #
external_styles = {
    "body": {"backgroundColor": "#f9f9f9", "color": "#333", "fontFamily": "Arial, sans-serif", "margin": "0", "padding": "0"},
    "sidebar": {"backgroundColor": "#fff", "padding": "20px", "borderRight": "2px solid #ddd", "height": "100vh", "overflowY": "auto"},
    "main": {"padding": "20px"}
}
app_css = {
    "container": {"display": "flex"},
    "sidebar": external_styles["sidebar"],
    "main": external_styles["body"],
    "tab": {"padding": "20px", "backgroundColor": "#fff", "border": "2px solid #ddd",
            "marginTop": "20px", "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.05)"}
}

dates = [datetime.today() - timedelta(days=i) for i in range(30)]
dates = sorted(dates)
kpi_history = pd.DataFrame({
    "Дата": dates,
    "Сукупна сума внесків": np.cumsum(np.random.randint(1000, 5000, size=30))
})
fig_kpi_history = px.line(kpi_history, x="Дата", y="Сукупна сума внесків",
                          title="Історія KPI: Сукупна сума внесків за останні 30 днів",
                          labels={"Сукупна сума внесків": "Сума внесків", "Дата": "Дата"})
fig_kpi_history.update_layout(**chart_config_default)

# ---------------------- Інтеграція аналізу волонтерської діяльності ---------------------- #
from analyze1 import VolunteerAnalysis
vol_analysis_instance = VolunteerAnalysis(data_dir="data")
fig_vol_scatter, fig_vol_topwords = vol_analysis_instance.create_plots(
    cluster_chart_config={
        "width": 1300,
        "height": 700,
        "title": "Альтернативна візуалізація кластеризації: Репутація vs. Тональність",
        "legend": {"title": {"text": "Кластер"}}
    },
    topwords_chart_config={
        "width": 1300,
        "height": 700,
        "title": "Альтернативний тематичний аналіз: Топ-10 ключових слів"
    }
)

# ---------------------- Інтеграція аналізу зборів та звітності (analyze2.py) ---------------------- #
from analyze2 import LevyAnalysis
levy_analysis_instance = LevyAnalysis(data_dir="data")
fig_levy_scatter, fig_levy_trend, fig_levy_corr = levy_analysis_instance.create_plots()

# ---------------------- Побудова Dash-додатку ---------------------- #
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(style=app_css["container"], children=[
    html.Div(style=app_css["sidebar"], children=[
        html.H2("Меню", style={"textAlign": "center", "color": "#4CAF50"}),
        html.Hr(),
        dcc.Tabs(id="sidebar-tabs", value="tab-analytics", children=[
            dcc.Tab(label="Аналітика", value="tab-analytics"),
            dcc.Tab(label="Зіркова схема", value="tab-star"),
            dcc.Tab(label="Сегментація донорів", value="tab-segmentation"),
            dcc.Tab(label="Прогнозування", value="tab-forecast"),
            dcc.Tab(label="Керування KPI", value="tab-kpi"),
            dcc.Tab(label="Запити", value="tab-requests"),
            dcc.Tab(label="Аналіз товарів", value="tab-products"),
            dcc.Tab(label="Волонтерський аналіз", value="tab-volunteers"),
            dcc.Tab(label="Аналіз зборів", value="tab-levy")
        ], vertical=True)
    ]),
    html.Div(id="main-content", style={"flex": "1", "padding": "20px"})
])

# ---------------------- Callback для перемикання контенту ---------------------- #
@app.callback(
    Output("main-content", "children"),
    Input("sidebar-tabs", "value")
)
def render_tab_content(tab_value):
    if tab_value == "tab-analytics":
        return html.Div([
            html.Div(style={"padding": "10px", "backgroundColor": "#fff", "border": "2px solid #ddd", "marginBottom": "20px"},
                     children=[
                         html.Label("Фільтр за роллю користувача:", style={"color": "#333"}),
                         dcc.Dropdown(
                             id="role-filter",
                             options=[{"label": role, "value": role} for role in sorted(users_df["user_role"].unique())],
                             value=[role for role in sorted(users_df["user_role"].unique())],
                             multi=True,
                             style={"border": "1px solid #ddd"}
                         ),
                         html.Label("Діапазон часу перебування (днів):", style={"color": "#333", "marginTop": "10px"}),
                         dcc.RangeSlider(
                             id="reg-duration-slider",
                             min=int(users_df["duration_on_platform"].min()),
                             max=int(users_df["duration_on_platform"].max()),
                             value=[int(users_df["duration_on_platform"].min()), int(users_df["duration_on_platform"].max())],
                             marks={i: str(i) for i in np.linspace(users_df["duration_on_platform"].min(), users_df["duration_on_platform"].max(), 10)}
                         )
                     ]),
            html.Div([
                html.H2("Середній внесок vs. Тривалість перебування", style={"color": "#333"}),
                dcc.Graph(id="scatter-graph", figure=fig_scatter)
            ]),
            html.Div([
                html.H2("Середній внесок по місяцях", style={"color": "#333"}),
                dcc.Graph(id="hist-donations", figure=fig_avg_donation_month)
            ]),
            html.Div([
                html.H2("Сукупна сума внесків по днях", style={"color": "#333"}),
                dcc.Graph(id="line-orders", figure=fig_line)
            ]),
            html.Div([
                html.H2("Кореляційна матриця", style={"color": "#333"}),
                dcc.Graph(id="heatmap", figure=fig_heatmap)
            ]),
            html.Div([
                html.H2("Розподіл часу перебування (дні)", style={"color": "#333"}),
                dcc.Graph(id="hist-reg", figure=fig_hist_reg)
            ]),
            html.Div(style={"border": "2px solid #ddd", "padding": "20px", "margin": "20px", "backgroundColor": "#fff"}, id="kpi-div-updated", children=[
                html.H2("Ключові показники (KPI)", style={"color": "#333"}),
                kpi_div
            ])
        ])
    elif tab_value == "tab-star":
        return html.Div(style=app_css["tab"], children=[
            html.H2("Зіркова схема OLAP‑куба", style={"color": "#333"}),
            cyto.Cytoscape(
                id='star-schema',
                elements=cyto_elements,
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '700px'},
                stylesheet=[
                    {'selector': 'node',
                     'style': {'label': 'data(label)',
                               'text-valign': 'center',
                               'color': '#333',
                               'background-color': '#4CAF50',
                               'width': 50,
                               'height': 50,
                               'font-size': 12,
                               'border-width': 2,
                               'border-color': '#ddd'}},
                    {'selector': 'node.fact',
                     'style': {'background-color': '#FF5722',
                               'width': 70,
                               'height': 70,
                               'font-size': 14}},
                    {'selector': 'edge',
                     'style': {'line-color': '#ccc',
                               'width': 3}}
                ]
            ),
            html.Div(id="drill-down", style={"padding": "20px", "marginTop": "20px", "border": "2px solid #ddd", "backgroundColor": "#fff"})
        ])
    elif tab_value == "tab-segmentation":
        return html.Div(style=app_css["tab"], children=[
            html.H2("Кластеризація донорів (K-Means)", style={"color": "#333"}),
            html.Label("Обрати кластер:", style={"color": "#333"}),
            dcc.Dropdown(
                id="cluster-filter",
                options=[{"label": f"Кластер {i}", "value": int(i)} for i in sorted(donation_stats["cluster"].unique())],
                value=[int(i) for i in sorted(donation_stats["cluster"].unique())],
                multi=True
            ),
            html.Div([dcc.Graph(id="cluster-scatter"), dcc.Graph(id="cluster-box")])
        ])
    elif tab_value == "tab-forecast":
        return html.Div(style=app_css["tab"], children=[
            html.H2("Прогнозування внесків", style={"color": "#333"}),
            html.Label("Горизонт прогнозу (днів):", style={"color": "#333"}),
            dcc.Slider(
                id="forecast-horizon-slider",
                min=30,
                max=180,
                step=10,
                value=90,
                marks={i: str(i) for i in range(30, 181, 10)}
            ),
            html.Div(id="forecast-output")
        ])
    elif tab_value == "tab-kpi":
        return html.Div(style=app_css["tab"], children=[
            kpi_controls,
            html.Div(id="kpi-div-updated", children=[html.H2("Ключові показники (KPI)", style={"color": "#333"}), kpi_div]),
            html.Div([html.H2("Історія KPI", style={"color": "#333"}), dcc.Graph(id="kpi-history", figure=fig_kpi_history)])
        ])
    elif tab_value == "tab-requests":
        return html.Div(style=app_css["tab"], children=[
            html.H2("Аналіз заявок за бригадами", style={"color": "#333"}),
            dcc.Graph(id="requests-analysis-graph", figure=fig_requests_by_brigade),
            html.Label("Оберіть метрику:", style={"color": "#333", "marginTop": "20px"}),
            dcc.Dropdown(
                id="request-metric-dropdown",
                options=[
                    {"label": "Кількість заявок", "value": "total_requests"},
                    {"label": "Сума заявок", "value": "total_amount"},
                    {"label": "Середня сума заявки", "value": "average_amount"}
                ],
                value="total_amount",
                style={"border": "1px solid #ddd"}
            )
        ])
    elif tab_value == "tab-products":
        return html.Div([
            html.H2("Аналіз заявок за товарами", style={"color": "#333"}),
            html.Div([
                html.Label("Фільтр за темою запиту:", style={"color": "#333"}),
                dcc.Input(
                    id="product-topic-filter",
                    type="text",
                    placeholder="Введіть тему запиту (наприклад, машини, тактична медицина)",
                    style={"width": "100%", "padding": "5px", "border": "1px solid #ddd"}
                )
            ], style={"margin": "10px"}),
            html.Div([dcc.Graph(id="product-bar-chart", figure=fig_products)]),
            html.Div([
                html.Label("Оберіть метрику:", style={"color": "#333", "marginTop": "20px"}),
                dcc.Dropdown(
                    id="product-metric-dropdown",
                    options=[
                        {"label": "Кількість заявок", "value": "request_count"},
                        {"label": "Загальна сума заявок", "value": "total_amount"},
                        {"label": "Середня сума заявки", "value": "average_amount"}
                    ],
                    value="request_count",
                    style={"border": "1px solid #ddd", "width": "50%"}
                )
            ], style={"margin": "10px"}),
            html.H2("Таблиця аналізу заявок за товарами", style={"color": "#333", "marginTop": "20px"}),
            product_table
        ], style=app_css["tab"])
    elif tab_value == "tab-volunteers":
        from analyze1 import VolunteerAnalysis
        va = VolunteerAnalysis(data_dir="data")
        fig_vol_scatter, fig_vol_topwords = va.create_plots(
            cluster_chart_config={
                "width": 1300,
                "height": 700,
                "title": "Альтернативна візуалізація кластеризації: Репутація vs. Тональність",
                "legend": {"title": {"text": "Кластер"}}
            },
            topwords_chart_config={
                "width": 1300,
                "height": 700,
                "title": "Альтернативний тематичний аналіз: Топ-10 ключових слів"
            }
        )
        return html.Div(style=app_css["tab"], children=[
            html.H2("Аналіз волонтерської діяльності та репутації", style={"color": "#333"}),
            html.H3("Кластеризація волонтерів за репутацією та тональністю", style={"color": "#333", "textAlign": "center"}),
            dcc.Graph(id="volunteer-scatter", figure=fig_vol_scatter),
            html.H3("Тематичний аналіз відгуків (топ-10 ключових слів)", style={"color": "#333", "textAlign": "center", "marginTop": "20px"}),
            dcc.Graph(id="volunteer-topwords", figure=fig_vol_topwords)
        ])
    elif tab_value == "tab-levy":
        from analyze2 import LevyAnalysis
        la = LevyAnalysis(data_dir="data")
        fig_levy_scatter, fig_levy_trend, fig_levy_corr = la.create_plots()
        return html.Div(style=app_css["tab"], children=[
            html.H2("Аналіз зборів та звітності", style={"color": "#333"}),
            dcc.Graph(id="levy-scatter", figure=fig_levy_scatter),
            dcc.Graph(id="levy-trend", figure=fig_levy_trend),
            dcc.Graph(id="levy-corr", figure=fig_levy_corr)
        ])
    else:
        return html.Div("Виберіть пункт меню.")

# ---------------------- Callback для оновлення графіків основної аналітики ---------------------- #
@app.callback(
    [Output("scatter-graph", "figure"),
     Output("hist-donations", "figure"),
     Output("hist-reg", "figure")],
    [Input("role-filter", "value"),
     Input("reg-duration-slider", "value")]
)
def update_main_analytics(selected_roles, reg_duration_range):
    filtered_stats = donation_stats[
        (donation_stats["user_role"].isin(selected_roles)) &
        (donation_stats["duration_on_platform"] >= reg_duration_range[0]) &
        (donation_stats["duration_on_platform"] <= reg_duration_range[1])
    ]
    updated_scatter = px.scatter(
        filtered_stats,
        x="duration_on_platform",
        y="avg_donation_user",
        color="user_role",
        title="Середній внесок vs. Тривалість перебування",
        labels={"duration_on_platform": "Тривалість (днів)", "avg_donation_user": "Середній внесок"}
    )
    updated_scatter.update_layout(**chart_config_default)
    liqpay_orders_df["month"] = liqpay_orders_df["create_date"].dt.to_period("M").dt.to_timestamp()
    monthly_avg = liqpay_orders_df.groupby("month")["amount"].mean().reset_index()
    updated_hist = px.bar(
        monthly_avg,
        x="month",
        y="amount",
        title="Середній внесок по місяцях",
        labels={"month": "Місяць", "amount": "Середній внесок"}
    )
    updated_hist.update_layout(**chart_config_default)
    updated_hist_reg = px.histogram(
        users_df[(users_df["duration_on_platform"] >= reg_duration_range[0]) &
                 (users_df["duration_on_platform"] <= reg_duration_range[1])],
        x="duration_on_platform",
        nbins=50,
        title="Розподіл часу перебування (дні)",
        labels={"duration_on_platform": "Тривалість (днів)"}
    )
    updated_hist_reg.update_layout(**chart_config_default)
    return updated_scatter, updated_hist, updated_hist_reg

@app.callback(
    Output("drill-down", "children"),
    Input("star-schema", "tapNodeData")
)
def drill_down(nodeData):
    if not nodeData:
        return "Натисніть на вузол для перегляду деталей."
    node_label = nodeData.get("label")
    if node_label == "User":
        fig = px.histogram(users_df, x="duration_on_platform", nbins=20, title="Деталізація: Тривалість перебування")
        return dcc.Graph(figure=fig, style={"width": "100%", "height": "600px"})
    elif node_label == "Request":
        try:
            req_df = load_data("request.csv")
            fig = px.histogram(req_df, x="amount", nbins=30, title="Деталізація: Суми заявок")
            return dcc.Graph(figure=fig, style={"width": "100%", "height": "600px"})
        except Exception as e:
            return f"Помилка: {str(e)}"
    elif node_label == "Liqpay_order":
        orders_by_day = liqpay_orders_df.groupby(liqpay_orders_df["create_date"].dt.date)["amount"].sum().reset_index()
        fig = px.line(orders_by_day, x="create_date", y="amount", title="Деталізація: Тренд транзакцій")
        return dcc.Graph(figure=fig, style={"width": "100%", "height": "600px"})
    else:
        return f"Вибрано: {node_label}. Деталі недоступні."

@app.callback(
    [Output("cluster-scatter", "figure"),
     Output("cluster-box", "figure")],
    Input("cluster-filter", "value")
)
def update_clusters(selected_clusters):
    filtered_clusters = donation_stats[donation_stats["cluster"].isin(selected_clusters)]
    fig_scatter_cluster = px.scatter(
        filtered_clusters,
        x="duration_on_platform",
        y="total_donations",
        color="cluster",
        title="Кластеризація: Тривалість vs. Сума внесків",
        labels={"duration_on_platform": "Тривалість (днів)", "total_donations": "Сума внесків", "cluster": "Кластер"}
    )
    fig_scatter_cluster.update_layout(**chart_config_default)
    fig_box_cluster = px.box(
        filtered_clusters,
        x="cluster",
        y="total_donations",
        title="Розподіл внесків за кластерами",
        labels={"cluster": "Кластер", "total_donations": "Сума внесків"}
    )
    fig_box_cluster.update_layout(**chart_config_default)
    return fig_scatter_cluster, fig_box_cluster

@app.callback(
    Output("forecast-output", "children"),
    Input("forecast-horizon-slider", "value")
)
def update_forecast(horizon):
    try:
        model = Prophet(daily_seasonality=True)
        model.fit(ts_df)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        fig_forecast = plot_plotly(model, forecast)
        fig_forecast.update_layout(title=f"Прогноз внесків на {horizon} днів", **chart_config_default)
        return dcc.Graph(figure=fig_forecast)
    except Exception as e:
        return html.P(f"Помилка прогнозування: {str(e)}", style={"color": "red"})

@app.callback(
    [Output("kpi-notifications", "children"),
     Output("kpi-div-updated", "children")],
    [Input("update-kpi-btn", "n_clicks")],
    [State("target-total-donations", "value"),
     State("target-avg-donation", "value"),
     State("target-unique-donors", "value"),
     State("target-percent-donors", "value"),
     State("target-max-donations", "value"),
     State("target-avg-reg-duration", "value"),
     State("target-corr", "value"),
     State("kpi-div-updated", "children")]
)
def update_kpi(n_clicks, total_target, avg_target, unique_target, percent_target, max_target, avg_reg_target, corr_target, kpi_content):
    if n_clicks == 0:
        return "", kpi_content
    new_targets = {
        "Загальна сума донатів": total_target,
        "Середній донат": avg_target,
        "Кількість унікальних донорів": unique_target,
        "Відсоток користувачів, що донатять": percent_target,
        "Максимальний донат": max_target,
        "Середня тривалість перебування на платформі (днів)": avg_reg_target,
        "Кореляція (перебування vs донати)": corr_target
    }
    new_kpi_info = calculate_kpi_info(new_targets)
    messages = []
    for kpi in new_kpi_info:
        if kpi["status"] == "Не досягнуто":
            messages.append(html.P(
                f"Попередження! {kpi['name']} не досягнуто: {kpi['actual']:.2f} vs {kpi['target']:.2f}.",
                style={"color": "red", "textAlign": "center"}
            ))
    if not messages:
        messages = html.P("Всі KPI відповідають встановленим цілям.", style={"color": "green", "textAlign": "center"})
    updated_kpi_div = create_kpi_div(new_kpi_info)
    new_content = html.Div(style={"border": "2px solid #ddd", "padding": "20px", "margin": "20px", "backgroundColor": "#fff"}, id="kpi-div-updated", children=[
        html.H2("Ключові показники (KPI)", style={"color": "#333"}),
        updated_kpi_div
    ])
    return messages, new_content

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
    }.get(metric, metric)
    fig = px.bar(
        grouped_requests,
        x="name",
        y=metric,
        title=f"Аналіз заявок: {y_label} за бригадами",
        labels={"name": "Бригада", metric: y_label}
    )
    fig.update_layout(**chart_config_default)
    return fig

@app.callback(
    Output("product-bar-chart", "figure"),
    [Input("product-metric-dropdown", "value"),
     Input("product-topic-filter", "value")]
)
def update_product_chart(metric, topic_filter):
    df = requests_df.copy()
    if topic_filter and topic_filter.strip() != "":
        df = df[df["description"].str.contains(topic_filter, case=False, na=False)]
    product_group = get_product_aggregation(df)
    y_label = {"request_count": "Кількість заявок", "total_amount": "Загальна сума заявок", "average_amount": "Середня сума заявки"}.get(metric, metric)
    fig = px.bar(product_group, x="product", y=metric,
                 title=f"Показник: {y_label} за товарами",
                 labels={"product": "Товар", metric: y_label})
    fig.update_layout(**chart_config_default)
    return fig

if __name__ == '__main__':
    app.run(debug=True)
