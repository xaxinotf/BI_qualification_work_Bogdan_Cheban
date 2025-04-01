import os
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_cytoscape as cyto

# ---------------------- Налаштування директорії та завантаження даних ---------------------- #
DATA_DIR = "data"

def load_data(filename):
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)

# Завантаження даних користувачів та транзакцій (донатів)
users_df = load_data("user.csv")
liqpay_orders_df = load_data("liqpay_order.csv")

# Використовуємо поле create_date як дату реєстрації (оскільки поля birth_date немає)
users_df["registration_date"] = pd.to_datetime(users_df["create_date"], errors="coerce")
users_df["registration_duration"] = (datetime.today() - users_df["registration_date"]).dt.days

liqpay_orders_df["create_date"] = pd.to_datetime(liqpay_orders_df["create_date"], errors="coerce")

# Об’єднання даних транзакцій із даними користувачів
donations = liqpay_orders_df.merge(users_df[["id", "registration_duration", "user_role"]],
                                   left_on="user_id", right_on="id", how="left")
donation_stats = donations.groupby("user_id", observed=False)["amount"].sum().reset_index().rename(
    columns={"amount": "total_donations"})
donation_stats = donation_stats.merge(users_df[["id", "registration_duration", "user_role"]],
                                      left_on="user_id", right_on="id", how="left")
corr_value = donation_stats["registration_duration"].corr(donation_stats["total_donations"])

# ---------------------- KPI Розрахунки ---------------------- #
total_donations = donation_stats["total_donations"].sum()
avg_donation = donation_stats["total_donations"].mean()
unique_donors = liqpay_orders_df["user_id"].nunique()
percent_donors = (unique_donors / len(users_df)) * 100
max_donation = donation_stats["total_donations"].max()
avg_reg_duration = donation_stats["registration_duration"].mean()

# Цільові значення для KPI (пороги – приклад)
kpi_targets = {
    "Загальна сума донатів": 1_000_000,
    "Середній донат": 200,
    "Кількість унікальних донорів": 5000,
    "Відсоток користувачів, що донатять": 10,
    "Максимальний донат": 100000,
    "Середня тривалість реєстрації (днів)": 365,
    "Кореляція (реєстрація vs донати)": 0.3
}

def evaluate_kpi(actual, target, higher_better=True):
    if higher_better:
        return "Досягнуто" if actual >= target else "Не досягнуто"
    else:
        return "Досягнуто" if actual <= target else "Не досягнуто"

kpi_info = [
    {"name": "Загальна сума донатів", "actual": total_donations, "target": kpi_targets["Загальна сума донатів"],
     "status": evaluate_kpi(total_donations, kpi_targets["Загальна сума донатів"], higher_better=True)},
    {"name": "Середній донат", "actual": avg_donation, "target": kpi_targets["Середній донат"],
     "status": evaluate_kpi(avg_donation, kpi_targets["Середній донат"], higher_better=True)},
    {"name": "Кількість унікальних донорів", "actual": unique_donors,
     "target": kpi_targets["Кількість унікальних донорів"],
     "status": evaluate_kpi(unique_donors, kpi_targets["Кількість унікальних донорів"], higher_better=True)},
    {"name": "Відсоток користувачів, що донатять", "actual": percent_donors,
     "target": kpi_targets["Відсоток користувачів, що донатять"],
     "status": evaluate_kpi(percent_donors, kpi_targets["Відсоток користувачів, що донатять"], higher_better=True)},
    {"name": "Максимальний донат", "actual": max_donation, "target": kpi_targets["Максимальний донат"],
     "status": evaluate_kpi(max_donation, kpi_targets["Максимальний донат"], higher_better=True)},
    {"name": "Середня тривалість реєстрації (днів)", "actual": avg_reg_duration,
     "target": kpi_targets["Середня тривалість реєстрації (днів)"],
     "status": evaluate_kpi(avg_reg_duration, kpi_targets["Середня тривалість реєстрації (днів)"],
                            higher_better=False)},
    {"name": "Кореляція (реєстрація vs донати)", "actual": corr_value,
     "target": kpi_targets["Кореляція (реєстрація vs донати)"],
     "status": evaluate_kpi(corr_value, kpi_targets["Кореляція (реєстрація vs донати)"], higher_better=True)}
]

# ---------------- Побудова графіків ---------------- #
fig_scatter = px.scatter(
    donation_stats,
    x="registration_duration",
    y="total_donations",
    color="user_role",
    title="Залежність сум донатів від тривалості реєстрації (днів)",
    labels={"registration_duration": "Тривалість реєстрації (днів)", "total_donations": "Сума донатів"}
)
fig_scatter.update_layout(height=600, width=1200)

fig_hist_donations = px.histogram(
    donation_stats,
    x="total_donations",
    nbins=50,
    title="Розподіл сум донатів користувачів",
    labels={"total_donations": "Сума донатів"}
)
fig_hist_donations.update_layout(height=600, width=1200)

fig_bar_roles = px.bar(
    users_df.groupby("user_role", observed=False).size().reset_index(name="кількість"),
    x="user_role",
    y="кількість",
    title="Кількість користувачів за ролями",
    labels={"user_role": "Роль", "кількість": "Кількість"}
)
fig_bar_roles.update_layout(height=600, width=1200)

orders_by_day = liqpay_orders_df.groupby(liqpay_orders_df["create_date"].dt.date)["amount"].sum().reset_index()
fig_line = px.line(
    orders_by_day,
    x="create_date",
    y="amount",
    title="Сукупна сума донатів по днях",
    labels={"create_date": "Дата", "amount": "Сума донатів"}
)
fig_line.update_layout(height=600, width=1200)

fig_pie_currency = px.pie(
    liqpay_orders_df,
    names="currency_name",
    title="Розподіл валют транзакцій"
)
fig_pie_currency.update_layout(height=600, width=1200)

numeric_cols = donation_stats.select_dtypes(include=[np.number]).columns
corr_matrix = donation_stats[numeric_cols].corr()
fig_heatmap = px.imshow(
    corr_matrix,
    text_auto=True,
    title="Матриця кореляції для числових показників"
)
fig_heatmap.update_layout(height=600, width=1200)

fig_hist_reg = px.histogram(
    users_df,
    x="registration_duration",
    nbins=50,
    title="Розподіл тривалості реєстрації користувачів (днів)",
    labels={"registration_duration": "Тривалість реєстрації (днів)"}
)
fig_hist_reg.update_layout(height=600, width=1200)

# ---------------- Побудова пивот-таблиці ---------------- #
donation_stats["reg_duration_bin"] = pd.cut(donation_stats["registration_duration"], bins=10)
pivot_table = donation_stats.pivot_table(
    index="user_role",
    columns="reg_duration_bin",
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
    sort_action="native"
)

# ---------------- Створення інтерактивної "зіркової схеми" OLAP‑куба ---------------- #
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
                html.H3(kpi["name"]),
                html.P(f"Фактичне значення: {value_str}"),
                html.P(f"Ціль: {target_str}"),
                html.P(f"Статус: {kpi['status']}")
            ], style={"border": "2px solid #ddd", "padding": "20px", "margin": "20px", "display": "inline-block",
                      "width": "30%", "backgroundColor": "#fff", "color": "#333"})
        )
    return html.Div(kpi_divs)

kpi_div = create_kpi_div(kpi_info)

# ---------------- Оформлення сторінки (CSS) ---------------- #
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

# ---------------- Створення Dash‑додатку з вкладками ---------------- #
role_options = [{"label": role, "value": role} for role in sorted(users_df["user_role"].unique())]
reg_duration_min = int(users_df["registration_duration"].min())
reg_duration_max = int(users_df["registration_duration"].max())
marks = {int(i): str(int(i)) for i in np.linspace(reg_duration_min, reg_duration_max, 10)}

app = dash.Dash(__name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])
app.layout = html.Div(style=external_styles["body"], children=[
    html.H1("BI-аналітика Волонтер+", style={"textAlign": "center", "padding": "20px", "backgroundColor": "#fff",
                                                 "borderBottom": "2px solid #ddd"}),
    dcc.Tabs([
        dcc.Tab(label="Основна аналітика", children=[
            html.Div(style={"padding": "10px", "border": "2px solid #ddd", "marginBottom": "20px",
                            "backgroundColor": "#fff"},
                     children=[
                         html.Label("Фільтр за роллю користувача:"),
                         dcc.Dropdown(
                             id="role-filter",
                             options=role_options,
                             value=[option["value"] for option in role_options],
                             multi=True
                         ),
                         html.Label("Діапазон тривалості реєстрації (днів):"),
                         dcc.RangeSlider(
                             id="reg-duration-slider",
                             min=reg_duration_min,
                             max=reg_duration_max,
                             value=[reg_duration_min, reg_duration_max],
                             marks=marks
                         )
                     ]),
            html.Div(style={"marginBottom": "20px"}, children=[
                html.H2("Графік: Сума донатів vs. Тривалість реєстрації"),
                dcc.Graph(id="scatter-graph", figure=fig_scatter, style={"width": 1200, "height": 600})
            ]),
            html.Div(style={"marginBottom": "20px"}, children=[
                html.H2("Гістограма: Розподіл сум донатів"),
                dcc.Graph(id="hist-donations", figure=fig_hist_donations, style={"width": 1200, "height": 600})
            ]),
            html.Div(style={"marginBottom": "20px"}, children=[
                html.H2("Стовпчикова діаграма: Кількість користувачів за ролями"),
                dcc.Graph(id="bar-roles", figure=fig_bar_roles, style={"width": 1200, "height": 600})
            ]),
            html.Div(style={"marginBottom": "20px"}, children=[
                html.H2("Лінійний графік: Сукупна сума донатів по днях"),
                dcc.Graph(id="line-orders", figure=fig_line, style={"width": 1200, "height": 600})
            ]),
            html.Div(style={"marginBottom": "20px"}, children=[
                html.H2("Кругова діаграма: Розподіл валют транзакцій"),
                dcc.Graph(id="pie-currency", figure=fig_pie_currency, style={"width": 1200, "height": 600})
            ]),
            html.Div(style={"marginBottom": "20px"}, children=[
                html.H2("Матриця кореляції для числових показників"),
                dcc.Graph(id="heatmap", figure=fig_heatmap, style={"width": 1200, "height": 600})
            ]),
            html.Div(style={"marginBottom": "20px"}, children=[
                html.H2("Гістограма: Розподіл тривалості реєстрації користувачів"),
                dcc.Graph(id="hist-reg", figure=fig_hist_reg, style={"width": 1200, "height": 600})
            ]),
            html.Div(
                style={"padding": "20px", "border": "2px solid #ddd", "marginTop": "20px", "backgroundColor": "#fff"},
                children=[
                    html.H2("Пивот-таблиця: Сума донатів за ролями та інтервалами реєстрації"),
                    pivot_table_div
                ]),
            html.Div(
                style={"padding": "20px", "border": "2px solid #ddd", "marginTop": "20px", "backgroundColor": "#fff"},
                children=[
                    html.H2("Ключові показники (KPIs)"),
                    kpi_div
                ])
        ]),
        dcc.Tab(label="Зіркова схема OLAP‑куба", children=[
            html.Div(
                style={"padding": "20px", "border": "2px solid #ddd", "marginTop": "20px", "backgroundColor": "#fff"},
                children=[
                    html.H2("Інтерактивна зіркова схема даних (OLAP‑куб)"),
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
        ])
    ])
])

# ---------------- Callbacks для оновлення графіків ---------------- #
@app.callback(
    [Output("scatter-graph", "figure"),
     Output("hist-donations", "figure"),
     Output("bar-roles", "figure"),
     Output("hist-reg", "figure")],
    [Input("role-filter", "value"),
     Input("reg-duration-slider", "value")]
)
def update_graphs(selected_roles, reg_duration_range):
    filtered_stats = donation_stats[
        (donation_stats["user_role"].isin(selected_roles)) &
        (donation_stats["registration_duration"] >= reg_duration_range[0]) &
        (donation_stats["registration_duration"] <= reg_duration_range[1])
    ]
    updated_scatter = px.scatter(
        filtered_stats,
        x="registration_duration",
        y="total_donations",
        color="user_role",
        title="Залежність сум донатів від тривалості реєстрації (днів)",
        labels={"registration_duration": "Тривалість реєстрації (днів)", "total_donations": "Сума донатів"}
    )
    updated_scatter.update_layout(height=600, width=1200)

    updated_hist = px.histogram(
        filtered_stats,
        x="total_donations",
        nbins=50,
        title="Розподіл сум донатів користувачів",
        labels={"total_donations": "Сума донатів"}
    )
    updated_hist.update_layout(height=600, width=1200)

    updated_bar = px.bar(
        users_df[users_df["user_role"].isin(selected_roles)]
        .groupby("user_role", observed=False).size().reset_index(name="кількість"),
        x="user_role",
        y="кількість",
        title="Кількість користувачів за ролями",
        labels={"user_role": "Роль", "кількість": "Кількість"}
    )
    updated_bar.update_layout(height=600, width=1200)

    updated_hist_reg = px.histogram(
        users_df[(users_df["registration_duration"] >= reg_duration_range[0]) &
                 (users_df["registration_duration"] <= reg_duration_range[1])],
        x="registration_duration",
        nbins=50,
        title="Розподіл тривалості реєстрації користувачів (днів)",
        labels={"registration_duration": "Тривалість реєстрації (днів)"}
    )
    updated_hist_reg.update_layout(height=600, width=1200)

    return updated_scatter, updated_hist, updated_bar, updated_hist_reg

# ---------------- Callback для drill‑down у зірковій схемі ---------------- #
@app.callback(
    Output("drill-down", "children"),
    Input("star-schema", "tapNodeData")
)
def drill_down(nodeData):
    if not nodeData:
        return "Натисніть на вузол для перегляду деталей."
    node_label = nodeData.get("label")
    if node_label == "User":
        fig = px.histogram(users_df, x="registration_duration", nbins=20, title="Деталізація: Тривалість реєстрації користувачів")
        return dcc.Graph(figure=fig, style={"width": 1200, "height": 600})
    elif node_label == "Request":
        try:
            requests_df = load_data("request.csv")
            fig = px.histogram(requests_df, x="amount", nbins=30, title="Деталізація: Розподіл сум запитів")
            return dcc.Graph(figure=fig, style={"width": 1200, "height": 600})
        except Exception as e:
            return f"Помилка завантаження даних запитів: {str(e)}"
    elif node_label == "Liqpay_order":
        orders_by_day = liqpay_orders_df.groupby(liqpay_orders_df["create_date"].dt.date)["amount"].sum().reset_index()
        fig = px.line(orders_by_day, x="create_date", y="amount", title="Деталізація: Тренд транзакцій")
        return dcc.Graph(figure=fig, style={"width": 1200, "height": 600})
    else:
        return f"Вибрано вимір: {node_label}. Детальна інформація не доступна."

# ---------------- Запуск додатку ---------------- #
if __name__ == '__main__':
    app.run(debug=True)
