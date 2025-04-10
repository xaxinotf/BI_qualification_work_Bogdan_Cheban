import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
from sklearn.cluster import KMeans
from prophet import Prophet
from prophet.plot import plot_plotly

# ---------------------- Налаштування директорії та завантаження даних ---------------------- #
DATA_DIR = "data"

def load_data(filename):
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)

# Завантаження даних із CSV
users_df = load_data("user.csv")
liqpay_orders_df = load_data("liqpay_order.csv")

# ---------------------- Обробка даних ---------------------- #
# Оскільки поля "birth_date" немає, використовуємо "create_date" як дату реєстрації
users_df["registration_date"] = pd.to_datetime(users_df["create_date"], errors="coerce")
# Розраховуємо тривалість перебування на платформі (в днях) та перейменовуємо стовпець
users_df["duration_on_platform"] = (datetime.today() - users_df["registration_date"]).dt.days

liqpay_orders_df["create_date"] = pd.to_datetime(liqpay_orders_df["create_date"], errors="coerce")

# Об’єднання транзакцій з даними користувачів
donations = liqpay_orders_df.merge(users_df[["id", "duration_on_platform", "user_role"]],
                                   left_on="user_id", right_on="id", how="left")
# Агрегація: сума донатів, кількість транзакцій та середній донат для кожного користувача
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

# Розрахунок кореляційної матриці
numeric_cols = donation_stats.select_dtypes(include=[np.number]).columns
corr_matrix = donation_stats[numeric_cols].corr()

# ---------------------- KPI Розрахунки ---------------------- #
total_donations = donation_stats["total_donations"].sum()
avg_donation = donation_stats["total_donations"].mean()  # загальне середнє значення
unique_donors = liqpay_orders_df["user_id"].nunique()
percent_donors = (unique_donors / len(users_df)) * 100
max_donation = donation_stats["total_donations"].max()
avg_duration_on_platform = donation_stats["duration_on_platform"].mean()

# Змінено ключі: використовується "перебування на платформі"
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
    if higher_better:
        return "Досягнуто" if actual >= target else "Не досягнуто"
    else:
        return "Досягнуто" if actual <= target else "Не досягнуто"

def calculate_kpi_info(targets):
    info = [
        {"name": "Загальна сума донатів", "actual": total_donations, "target": targets["Загальна сума донатів"],
         "status": evaluate_kpi(total_donations, targets["Загальна сума донатів"], higher_better=True)},
        {"name": "Середній донат", "actual": avg_donation, "target": targets["Середній донат"],
         "status": evaluate_kpi(avg_donation, targets["Середній донат"], higher_better=True)},
        {"name": "Кількість унікальних донорів", "actual": unique_donors,
         "target": targets["Кількість унікальних донорів"],
         "status": evaluate_kpi(unique_donors, targets["Кількість унікальних донорів"], higher_better=True)},
        {"name": "Відсоток користувачів, що донатять", "actual": percent_donors,
         "target": targets["Відсоток користувачів, що донатять"],
         "status": evaluate_kpi(percent_donors, targets["Відсоток користувачів, що донатять"], higher_better=True)},
        {"name": "Максимальний донат", "actual": max_donation, "target": targets["Максимальний донат"],
         "status": evaluate_kpi(max_donation, targets["Максимальний донат"], higher_better=True)},
        {"name": "Середня тривалість перебування на платформі (днів)", "actual": avg_duration_on_platform,
         "target": targets["Середня тривалість перебування на платформі (днів)"],
         "status": evaluate_kpi(avg_duration_on_platform, targets["Середня тривалість перебування на платформі (днів)"],
                                higher_better=False)},
        {"name": "Кореляція (перебування vs донати)", "actual": corr_value,
         "target": targets["Кореляція (перебування vs донати)"],
         "status": evaluate_kpi(corr_value, targets["Кореляція (перебування vs донати)"], higher_better=True)}
    ]
    return info

kpi_info = calculate_kpi_info(default_kpi_targets)

# ---------------- Сегментація та кластеризація донорів (K‑Means) ---------------- #
X = donation_stats[["duration_on_platform", "total_donations"]].fillna(0)
kmeans = KMeans(n_clusters=4, random_state=42)
donation_stats["cluster"] = kmeans.fit_predict(X)

# ---------------- Побудова графіків ---------------- #

# (3) Графік: Середній донат vs. Тривалість перебування на платформі
fig_scatter = px.scatter(
    donation_stats,
    x="duration_on_platform",
    y="avg_donation_user",
    color="user_role",
    title="Середній донат vs. Тривалість перебування на платформі",
    labels={"duration_on_platform": "Тривалість перебування на платформі (днів)",
            "avg_donation_user": "Середній донат"}
)
fig_scatter.update_layout(height=600, width=1200)

# (4) Гістограма: Середній донат по місяцях
liqpay_orders_df["month"] = liqpay_orders_df["create_date"].dt.to_period("M").dt.to_timestamp()
monthly_avg = liqpay_orders_df.groupby("month")["amount"].mean().reset_index()
fig_avg_donation_month = px.bar(
    monthly_avg,
    x="month",
    y="amount",
    title="Середній донат по місяцях",
    labels={"month": "Місяць", "amount": "Середній донат"}
)
fig_avg_donation_month.update_layout(height=600, width=1200)

# (1) Матриця кореляції для числових показників з перейменуванням осей і описом
fig_heatmap = px.imshow(
    corr_matrix,
    text_auto=True,
    title="Матриця кореляції числових показників (з описом)"
)
fig_heatmap.update_xaxes(title_text="Показники (X)")
fig_heatmap.update_yaxes(title_text="Показники (Y)")
fig_heatmap.add_annotation(
    text="Ця матриця показує силу та напрямок зв'язків між числовими показниками.",
    xref="paper", yref="paper",
    x=0.5, y=-0.15, showarrow=False,
    font=dict(size=12, color="#666")
)
fig_heatmap.update_layout(height=600, width=1200)

# (2) Гістограма: Розподіл тривалості перебування на платформі (днів)
fig_hist_reg = px.histogram(
    users_df,
    x="duration_on_platform",
    nbins=50,
    title="Розподіл тривалості перебування на платформі (днів)",
    labels={"duration_on_platform": "Тривалість перебування на платформі (днів)"}
)
fig_hist_reg.update_layout(height=600, width=1200)

# Лінійний графік для транзакцій
orders_by_day = liqpay_orders_df.groupby(liqpay_orders_df["create_date"].dt.date)["amount"].sum().reset_index()
fig_line = px.line(
    orders_by_day,
    x="create_date",
    y="amount",
    title="Сукупна сума донатів по днях",
    labels={"create_date": "Дата", "amount": "Сума донатів"}
)
fig_line.update_layout(height=600, width=1200)

# ---------------- Побудова пивот-таблиці ---------------- #
donation_stats["duration_bin"] = pd.cut(donation_stats["duration_on_platform"], bins=10)
pivot_table = donation_stats.pivot_table(
    index="user_role",
    columns="duration_bin",
    values="total_donations",
    aggfunc="sum",
    fill_value=0
).reset_index()
pivot_table.columns = pivot_table.columns.map(str)
pivot_table_div = dash_table.DataTable(
    id='pivot-table',
    columns=[{"name": str(col), "id": str(col)} for col in pivot_table.columns],
    data=pivot_table.to_dict('records'),
    style_table={'overflowX': 'auto'},
    page_size=10,
    filter_action="native",
    sort_action="native",
    style_cell={'textAlign': 'center', 'padding': '5px'},
    style_header={'backgroundColor': '#ddd', 'fontWeight': 'bold'}
)

# ---------------- Побудова інтерактивної "зіркової схеми" OLAP‑куба ---------------- #
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

# ---------------- Блок KPI ---------------- #
def create_kpi_div(kpi_info):
    kpi_divs = []
    for kpi in kpi_info:
        value_str = f"{kpi['actual']:.2f}" if isinstance(kpi["actual"], (float, int)) else f"{kpi['actual']}"
        target_str = f"{kpi['target']:.2f}" if isinstance(kpi["target"], (float, int)) else f"{kpi['target']}"
        kpi_divs.append(
            html.Div([
                html.H3(kpi["name"], style={"color": "#333"}),
                html.P(f"Фактичне значення: {value_str}", style={"color": "#555"}),
                html.P(f"Ціль: {target_str}", style={"color": "#4CAF50"}),
                html.P(f"Статус: {kpi['status']}", style={"color": "#666"})
            ], style={
                "border": "2px solid #ddd", "padding": "20px", "margin": "20px",
                "display": "inline-block", "width": "30%", "backgroundColor": "#fff",
                "color": "#333", "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.05)"
            })
        )
    return html.Div(kpi_divs)

kpi_div = create_kpi_div(kpi_info)

# ---------------- Прогнозування трендів із використанням Prophet ---------------- #
ts_df = liqpay_orders_df.groupby(liqpay_orders_df["create_date"].dt.date)["amount"].sum().reset_index()
ts_df.columns = ["ds", "y"]
ts_df["ds"] = pd.to_datetime(ts_df["ds"])

def forecast_donations(horizon_days):
    model = Prophet(daily_seasonality=True)
    model.fit(ts_df)
    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)
    fig_forecast = plot_plotly(model, forecast)
    fig_forecast.update_layout(
        title=f"Прогноз сукупної суми донатів на наступні {horizon_days} днів",
        height=600, width=1200
    )
    return fig_forecast

# ---------------- Панель управління KPI (з можливістю коригування порогів та перегляду сповіщень) ---------------- #
kpi_controls = html.Div([
    html.H2("Налаштування порогів KPI", style={"color": "#333"}),
    html.Div([
        html.Label("Загальна сума донатів (ціль):", style={"color": "#333"}),
        dcc.Input(id="target-total-donations", type="number", value=default_kpi_targets["Загальна сума донатів"],
                  style={"border": "1px solid #ddd", "padding": "5px"})
    ], style={"margin": "10px"}),
    html.Div([
        html.Label("Середній донат (ціль):", style={"color": "#333"}),
        dcc.Input(id="target-avg-donation", type="number", value=default_kpi_targets["Середній донат"],
                  style={"border": "1px solid #ddd", "padding": "5px"})
    ], style={"margin": "10px"}),
    html.Div([
        html.Label("Кількість унікальних донорів (ціль):", style={"color": "#333"}),
        dcc.Input(id="target-unique-donors", type="number", value=default_kpi_targets["Кількість унікальних донорів"],
                  style={"border": "1px solid #ddd", "padding": "5px"})
    ], style={"margin": "10px"}),
    html.Div([
        html.Label("Відсоток користувачів, що донатять (ціль):", style={"color": "#333"}),
        dcc.Input(id="target-percent-donors", type="number", value=default_kpi_targets["Відсоток користувачів, що донатять"],
                  style={"border": "1px solid #ddd", "padding": "5px"})
    ], style={"margin": "10px"}),
    html.Div([
        html.Label("Максимальний донат (ціль):", style={"color": "#333"}),
        dcc.Input(id="target-max-donation", type="number", value=default_kpi_targets["Максимальний донат"],
                  style={"border": "1px solid #ddd", "padding": "5px"})
    ], style={"margin": "10px"}),
    html.Div([
        html.Label("Середня тривалість перебування на платформі (днів, ціль):", style={"color": "#333"}),
        dcc.Input(id="target-avg-reg-duration", type="number", value=default_kpi_targets["Середня тривалість перебування на платформі (днів)"],
                  style={"border": "1px solid #ddd", "padding": "5px"})
    ], style={"margin": "10px"}),
    html.Div([
        html.Label("Кореляція (ціль):", style={"color": "#333"}),
        dcc.Input(id="target-corr", type="number", value=default_kpi_targets["Кореляція (перебування vs донати)"], step=0.01,
                  style={"border": "1px solid #ddd", "padding": "5px"})
    ], style={"margin": "10px"}),
    html.Button("Оновити пороги KPI", id="update-kpi-btn", n_clicks=0,
                style={"backgroundColor": "#4CAF50", "color": "#fff", "padding": "10px 20px",
                       "border": "none", "cursor": "pointer", "margin": "10px"}),
    html.Div(id="kpi-notifications", style={"marginTop": "20px", "color": "#333"})
], style={"border": "2px solid #ddd", "padding": "20px", "margin": "20px", "backgroundColor": "#fff"})

# ---------------- Додаткове оформлення (CSS) ---------------- #
external_styles = {
    "body": {
        "backgroundColor": "#f9f9f9",
        "color": "#333",
        "fontFamily": "Arial, sans-serif",
        "margin": "0",
        "padding": "0"
    },
    "header": {
        "backgroundColor": "#fff",
        "borderBottom": "2px solid #ddd",
        "padding": "10px"
    },
    "card": {
        "backgroundColor": "#fff",
        "border": "1px solid #ddd",
        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.05)",
        "padding": "20px",
        "margin": "20px"
    },
    "button": {
        "backgroundColor": "#4CAF50",
        "color": "#fff",
        "border": "none",
        "padding": "10px 20px",
        "cursor": "pointer",
        "transition": "background-color 0.3s"
    },
    "button_hover": {
        "backgroundColor": "#45a049"
    }
}

# ---------------- Додаткове оформлення контейнера ---------------- #
app_css = {
    "container": {
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "20px"
    },
    "tab": {
        "padding": "20px",
        "backgroundColor": "#fff",
        "border": "2px solid #ddd",
        "marginTop": "20px",
        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.05)"
    }
}

# ---------------- Додаткова вкладка: Історія KPI (симульована) ---------------- #
dates = [datetime.today() - timedelta(days=i) for i in range(30)]
dates = sorted(dates)
kpi_history = pd.DataFrame({
    "Дата": dates,
    "Сукупна сума донатів": np.cumsum(np.random.randint(1000, 5000, size=30))
})
fig_kpi_history = px.line(kpi_history, x="Дата", y="Сукупна сума донатів",
                          title="Історія KPI: Сукупна сума донатів за останні 30 днів",
                          labels={"Сукупна сума донатів": "Сума донатів", "Дата": "Дата"})
fig_kpi_history.update_layout(height=600, width=1200)

# ---------------- Створення додатку з вкладками ---------------- #
role_options = [{"label": role, "value": role} for role in sorted(users_df["user_role"].unique())]
reg_duration_min = int(users_df["duration_on_platform"].min())
reg_duration_max = int(users_df["duration_on_platform"].max())
marks = {int(i): str(int(i)) for i in np.linspace(reg_duration_min, reg_duration_max, 10)}

app = dash.Dash(__name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])

app.layout = html.Div(style=app_css["container"], children=[
    html.H1("BI-аналітика Волонтер+", style={"textAlign": "center", "padding": "20px", "backgroundColor": "#fff",
                                                 "borderBottom": "2px solid #ddd"}),
    dcc.Tabs([
        dcc.Tab(label="Основна аналітика", children=[
            html.Div(style={"padding": "10px", "backgroundColor": "#fff", "border": "2px solid #ddd", "marginBottom": "20px"},
                     children=[
                         html.Label("Фільтр за роллю користувача:", style={"color": "#333"}),
                         dcc.Dropdown(
                             id="role-filter",
                             options=role_options,
                             value=[option["value"] for option in role_options],
                             multi=True,
                             style={"border": "1px solid #ddd"}
                         ),
                         html.Label("Діапазон тривалості перебування на платформі (днів):", style={"color": "#333"}),
                         dcc.RangeSlider(
                             id="reg-duration-slider",
                             min=reg_duration_min,
                             max=reg_duration_max,
                             value=[reg_duration_min, reg_duration_max],
                             marks=marks
                         )
                     ]),
            html.Div(style={"marginBottom": "20px"}, children=[
                html.H2("Графік: Середній донат vs. Тривалість перебування на платформі", style={"color": "#333"}),
                dcc.Graph(id="scatter-graph", figure=fig_scatter, style={"width": 1200, "height": 600})
            ]),
            html.Div(style={"marginBottom": "20px"}, children=[
                html.H2("Гістограма: Середній донат по місяцях", style={"color": "#333"}),
                dcc.Graph(id="hist-donations", figure=fig_avg_donation_month, style={"width": 1200, "height": 600})
            ]),
            html.Div(style={"marginBottom": "20px"}, children=[
                html.H2("Лінійний графік: Сукупна сума донатів по днях", style={"color": "#333"}),
                dcc.Graph(id="line-orders", figure=fig_line, style={"width": 1200, "height": 600})
            ]),
            html.Div(style={"marginBottom": "20px"}, children=[
                html.H2("Матриця кореляції для числових показників", style={"color": "#333"}),
                dcc.Graph(id="heatmap", figure=fig_heatmap, style={"width": 1200, "height": 600})
            ]),
            html.Div(style={"marginBottom": "20px"}, children=[
                html.H2("Гістограма: Розподіл тривалості перебування на платформі (днів)", style={"color": "#333"}),
                dcc.Graph(id="hist-reg", figure=fig_hist_reg, style={"width": 1200, "height": 600})
            ]),
            html.Div(style=app_css["tab"], children=[
                html.H2("Пивот-таблиця: Сума донатів за ролями та інтервалами перебування на платформі", style={"color": "#333"}),
                pivot_table_div
            ]),
            html.Div(style=app_css["tab"], id="kpi-div-updated", children=[
                html.H2("Ключові показники (KPIs)", style={"color": "#333"}),
                kpi_div
            ])
        ]),
        dcc.Tab(label="Зіркова схема OLAP‑куба", children=[
            html.Div(style=app_css["tab"], children=[
                html.H2("Інтерактивна зіркова схема даних (OLAP‑куб)", style={"color": "#333"}),
                cyto.Cytoscape(
                    id='star-schema',
                    elements=cyto_elements,
                    layout={'name': 'preset'},
                    style={'width': 1200, 'height': 800},
                    stylesheet=[
                        {
                            'selector': 'node',
                            'style': {
                                'label': 'data(label)',
                                'text-valign': 'center',
                                'color': '#333',
                                'background-color': '#4CAF50',
                                'width': 50,
                                'height': 50,
                                'font-size': 12,
                                'border-width': 2,
                                'border-color': '#ddd'
                            }
                        },
                        {
                            'selector': 'node.fact',
                            'style': {
                                'background-color': '#FF5722',
                                'width': 70,
                                'height': 70,
                                'font-size': 14
                            }
                        },
                        {
                            'selector': 'edge',
                            'style': {
                                'line-color': '#ccc',
                                'width': 3
                            }
                        }
                    ]
                ),
                html.Div(id="drill-down", style={"padding": "20px", "marginTop": "20px", "border": "2px solid #ddd", "backgroundColor": "#fff"})
            ])
        ]),
        dcc.Tab(label="Сегментація донорів", children=[
            html.Div(style=app_css["tab"], children=[
                html.H2("Сегментація та кластеризація донорів (K-Means)", style={"color": "#333"}),
                html.Label("Виберіть кластер:", style={"color": "#333"}),
                dcc.Dropdown(
                    id="cluster-filter",
                    options=[{"label": f"Кластер {i}", "value": int(i)} for i in sorted(donation_stats["cluster"].unique())],
                    value=[int(i) for i in sorted(donation_stats["cluster"].unique())],
                    multi=True
                ),
                html.Div([
                    dcc.Graph(id="cluster-scatter", style={"width": 1200, "height": 600}),
                    dcc.Graph(id="cluster-box", style={"width": 1200, "height": 600})
                ])
            ])
        ]),
        dcc.Tab(label="Прогнозування трендів", children=[
            html.Div(style=app_css["tab"], children=[
                html.H2("Прогнозування сукупної суми донатів", style={"color": "#333"}),
                html.Label("Виберіть горизонт прогнозування (днів):", style={"color": "#333"}),
                dcc.Slider(
                    id="forecast-horizon-slider",
                    min=30,
                    max=180,
                    step=10,
                    value=90,
                    marks={i: str(i) for i in range(30, 181, 10)}
                ),
                html.Div(id="forecast-output", style={"marginTop": "20px"})
            ])
        ]),
        dcc.Tab(label="Контроль KPI", children=[
            html.Div(style=app_css["tab"], children=[
                kpi_controls,
                html.Div(style={"marginTop": "20px"}, children=[
                    html.H2("Історія KPI", style={"color": "#333"}),
                    dcc.Graph(id="kpi-history", figure=fig_kpi_history, style={"width": 1200, "height": 600})
                ])
            ])
        ])
    ])
])

# ---------------- Callback для оновлення основних графіків ---------------- #
@app.callback(
    [Output("scatter-graph", "figure"),
     Output("hist-donations", "figure"),
     Output("hist-reg", "figure")],
    [Input("role-filter", "value"),
     Input("reg-duration-slider", "value")]
)
def update_graphs(selected_roles, reg_duration_range):
    filtered_stats = donation_stats[
        (donation_stats["user_role"].isin(selected_roles)) &
        (donation_stats["duration_on_platform"] >= reg_duration_range[0]) &
        (donation_stats["duration_on_platform"] <= reg_duration_range[1])
    ]
    # Середній донат vs. тривалість перебування
    updated_scatter = px.scatter(
        filtered_stats,
        x="duration_on_platform",
        y="avg_donation_user",
        color="user_role",
        title="Середній донат vs. Тривалість перебування на платформі",
        labels={"duration_on_platform": "Тривалість перебування на платформі (днів)",
                "avg_donation_user": "Середній донат"}
    )
    updated_scatter.update_layout(height=600, width=1200)

    # Середній донат по місяцях
    liqpay_orders_df["month"] = liqpay_orders_df["create_date"].dt.to_period("M").dt.to_timestamp()
    monthly_avg = liqpay_orders_df.groupby("month")["amount"].mean().reset_index()
    updated_hist = px.bar(
        monthly_avg,
        x="month",
        y="amount",
        title="Середній донат по місяцях",
        labels={"month": "Місяць", "amount": "Середній донат"}
    )
    updated_hist.update_layout(height=600, width=1200)

    # Гістограма: Розподіл тривалості перебування
    updated_hist_reg = px.histogram(
        users_df[(users_df["duration_on_platform"] >= reg_duration_range[0]) &
                 (users_df["duration_on_platform"] <= reg_duration_range[1])],
        x="duration_on_platform",
        nbins=50,
        title="Розподіл тривалості перебування на платформі (днів)",
        labels={"duration_on_platform": "Тривалість перебування на платформі (днів)"}
    )
    updated_hist_reg.update_layout(height=600, width=1200)

    return updated_scatter, updated_hist, updated_hist_reg

# ---------------- Callback для drill‑down у зірковій схемі OLAP‑куба ---------------- #
@app.callback(
    Output("drill-down", "children"),
    Input("star-schema", "tapNodeData")
)
def drill_down(nodeData):
    if not nodeData:
        return "Натисніть на вузол для перегляду деталей."
    node_label = nodeData.get("label")
    if node_label == "User":
        fig = px.histogram(users_df, x="duration_on_platform", nbins=20, title="Деталізація: Тривалість перебування на платформі")
        return dcc.Graph(figure=fig, style={"width": 1200, "height": 600})
    elif node_label == "Request":
        try:
            req_df = load_data("request.csv")
            fig = px.histogram(req_df, x="amount", nbins=30, title="Деталізація: Розподіл сум запитів")
            return dcc.Graph(figure=fig, style={"width": 1200, "height": 600})
        except Exception as e:
            return f"Помилка завантаження даних запитів: {str(e)}"
    elif node_label == "Liqpay_order":
        orders_by_day = liqpay_orders_df.groupby(liqpay_orders_df["create_date"].dt.date)["amount"].sum().reset_index()
        fig = px.line(orders_by_day, x="create_date", y="amount", title="Деталізація: Тренд транзакцій")
        return dcc.Graph(figure=fig, style={"width": 1200, "height": 600})
    else:
        return f"Вибрано вимір: {node_label}. Детальна інформація не доступна."

# ---------------- Callback для кластеризації донорів ---------------- #
@app.callback(
    [Output("cluster-scatter", "figure"),
     Output("cluster-box", "figure")],
    Input("cluster-filter", "value")
)
def update_cluster_graphs(selected_clusters):
    filtered_clusters = donation_stats[donation_stats["cluster"].isin(selected_clusters)]
    fig_scatter_cluster = px.scatter(
        filtered_clusters,
        x="duration_on_platform",
        y="total_donations",
        color="cluster",
        title="Сегментація донорів: Тривалість перебування на платформі vs Сума донатів",
        labels={"duration_on_platform": "Тривалість перебування на платформі (днів)",
                "total_donations": "Сума донатів", "cluster": "Кластер"}
    )
    fig_scatter_cluster.update_layout(height=600, width=1200)

    fig_box_cluster = px.box(
        filtered_clusters,
        x="cluster",
        y="total_donations",
        title="Розподіл сум донатів за кластерами",
        labels={"cluster": "Кластер", "total_donations": "Сума донатів"}
    )
    fig_box_cluster.update_layout(height=600, width=1200)
    return fig_scatter_cluster, fig_box_cluster

# ---------------- Callback для прогнозування трендів донатів ---------------- #
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
        fig_forecast.update_layout(title=f"Прогноз сукупної суми донатів на наступні {horizon} днів",
                                   height=600, width=1200)
        return dcc.Graph(figure=fig_forecast, style={"width": 1200, "height": 600})
    except Exception as e:
        return html.P(f"Помилка прогнозування: {str(e)}", style={"color": "red"})

# ---------------- Callback для оновлення KPI порогів та сповіщень ---------------- #
@app.callback(
    [Output("kpi-notifications", "children"),
     Output("kpi-div-updated", "children")],
    [Input("update-kpi-btn", "n_clicks")],
    [State("target-total-donations", "value"),
     State("target-avg-donation", "value"),
     State("target-unique-donors", "value"),
     State("target-percent-donors", "value"),
     State("target-max-donation", "value"),
     State("target-avg-reg-duration", "value"),
     State("target-corr", "value")]
)
def update_kpi_thresholds(n_clicks, total_target, avg_target, unique_target, percent_target, max_target, avg_reg_target, corr_target):
    if n_clicks == 0:
        return "", ""
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
                f"Попередження! {kpi['name']} не досягнуто: фактичне значення {kpi['actual']:.2f} менше цільового {kpi['target']:.2f}.",
                style={"color": "red"}
            ))
    if not messages:
        messages = html.P("Всі KPI відповідають встановленим цілям.", style={"color": "green"})
    updated_kpi_div = create_kpi_div(new_kpi_info)
    return messages, updated_kpi_div

# ---------------- Запуск додатку ---------------- #
if __name__ == '__main__':
    app.run(debug=True)
