import os
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

# Директорія з CSV-файлами
DATA_DIR = "data"


def load_data(filename):
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)


# Завантаження даних користувачів та транзакцій (донатів)
users_df = load_data("user.csv")
liqpay_orders_df = load_data("liqpay_order.csv")

# Оскільки в user.csv немає поля "birth_date", використаємо "create_date" як дату реєстрації
users_df["registration_date"] = pd.to_datetime(users_df["create_date"], errors="coerce")
# Обчислюємо тривалість реєстрації (кількість днів від дати реєстрації до сьогодні)
users_df["registration_duration"] = (datetime.today() - users_df["registration_date"]).dt.days

# Перетворення дат у liqpay_order.csv
liqpay_orders_df["create_date"] = pd.to_datetime(liqpay_orders_df["create_date"], errors="coerce")

# Об’єднання даних донатів із даними користувачів
donations = liqpay_orders_df.merge(users_df[["id", "registration_duration", "user_role"]],
                                   left_on="user_id", right_on="id", how="left")

# Агрегування: сума донатів по кожному користувачу
donation_stats = donations.groupby("user_id")["amount"].sum().reset_index().rename(
    columns={"amount": "total_donations"})
donation_stats = donation_stats.merge(users_df[["id", "registration_duration", "user_role"]],
                                      left_on="user_id", right_on="id", how="left")

# Обчислення кореляції між тривалістю реєстрації та сумою донатів
corr_value = donation_stats["registration_duration"].corr(donation_stats["total_donations"])

# ---------------- KPI Розрахунки ----------------
total_donations = donation_stats["total_donations"].sum()
avg_donation = donation_stats["total_donations"].mean()
unique_donors = liqpay_orders_df["user_id"].nunique()
percent_donors = (unique_donors / len(users_df)) * 100
max_donation = donation_stats["total_donations"].max()
avg_reg_duration = donation_stats["registration_duration"].mean()

# Встановлюємо цільові значення для KPI (target) – за замовчуванням використовуємо деякі порогові значення
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

# ---------------- Побудова графіків ----------------
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
    users_df.groupby("user_role").size().reset_index(name="кількість"),
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

# ---------------- Побудова пивот-таблиці ----------------
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


# ---------------- Побудова "зіркової схеми" ----------------
def create_star_schema_diagram():
    fact = "Liqpay_order"
    dimensions = ["User", "Volunteer", "Military Personnel", "Request", "Levy",
                  "Volunteer_Levy", "Report", "Attachment", "Brigade Codes", "Add Request",
                  "Email Template", "Email Notification", "Email Recipient", "Email Attachment", "AI Chat Messages"]
    n_dim = len(dimensions)
    x_fact, y_fact = 0, 0
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False)
    x_dims = 4 * np.cos(angles)
    y_dims = 4 * np.sin(angles)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[x_fact],
        y=[y_fact],
        mode="markers+text",
        marker=dict(size=30, color="red"),
        text=[fact],
        textposition="middle center",
        name="Факт"
    ))
    for i, dim in enumerate(dimensions):
        fig.add_trace(go.Scatter(
            x=[x_dims[i]],
            y=[y_dims[i]],
            mode="markers+text",
            marker=dict(size=25, color="blue"),
            text=[dim],
            textposition="top center",
            name="Вимір"
        ))
        fig.add_shape(
            type="line",
            x0=x_fact, y0=y_fact, x1=x_dims[i], y1=y_dims[i],
            line=dict(color="gray", width=3)
        )
    fig.update_layout(
        title="Зіркова схема даних",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=800,
        width=1200
    )
    return fig


fig_star = create_star_schema_diagram()


# ---------------- Побудова блоку KPI ----------------
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
            ], style={"border": "2px solid #ccc", "padding": "20px", "margin": "20px", "display": "inline-block",
                      "width": "30%"})
        )
    return html.Div(kpi_divs)


kpi_div = create_kpi_div(kpi_info)

# ---------------- Створення Dash‑додатку з вкладками ----------------
role_options = [{"label": role, "value": role} for role in sorted(users_df["user_role"].unique())]
reg_duration_min = int(users_df["registration_duration"].min())
reg_duration_max = int(users_df["registration_duration"].max())
marks = {int(i): str(int(i)) for i in np.linspace(reg_duration_min, reg_duration_max, 10)}

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("BI-аналітика Волонтер+"),
    dcc.Tabs([
        dcc.Tab(label="Основна аналітика", children=[
            html.Div([
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
            ], style={"padding": "10px", "border": "2px solid #ccc", "margin-bottom": "20px"}),

            html.Div([
                html.H2("Графік: Сума донатів vs. Тривалість реєстрації"),
                dcc.Graph(id="scatter-graph", figure=fig_scatter, style={"width": "1200px", "height": "600px"})
            ]),

            html.Div([
                html.H2("Гістограма: Розподіл сум донатів"),
                dcc.Graph(id="hist-donations", figure=fig_hist_donations, style={"width": "1200px", "height": "600px"})
            ]),

            html.Div([
                html.H2("Стовпчикова діаграма: Кількість користувачів за ролями"),
                dcc.Graph(id="bar-roles", figure=fig_bar_roles, style={"width": "1200px", "height": "600px"})
            ]),

            html.Div([
                html.H2("Лінійний графік: Сукупна сума донатів по днях"),
                dcc.Graph(id="line-orders", figure=fig_line, style={"width": "1200px", "height": "600px"})
            ]),

            html.Div([
                html.H2("Кругова діаграма: Розподіл валют транзакцій"),
                dcc.Graph(id="pie-currency", figure=fig_pie_currency, style={"width": "1200px", "height": "600px"})
            ]),

            html.Div([
                html.H2("Матриця кореляції для числових показників"),
                dcc.Graph(id="heatmap", figure=fig_heatmap, style={"width": "1200px", "height": "600px"})
            ]),

            html.Div([
                html.H2("Гістограма: Розподіл тривалості реєстрації користувачів"),
                dcc.Graph(id="hist-reg", figure=fig_hist_reg, style={"width": "1200px", "height": "600px"})
            ]),

            html.Div([
                html.H2("Пивот-таблиця: Сума донатів за ролями та інтервалами реєстрації"),
                pivot_table_div
            ], style={"padding": "20px", "border": "2px solid #ccc", "margin-top": "20px"}),

            html.Div([
                html.H2("Ключові показники (KPIs)"),
                kpi_div
            ], style={"padding": "20px", "border": "2px solid #ccc", "margin-top": "20px"})
        ]),
        dcc.Tab(label="Зіркова схема", children=[
            html.Div([
                html.H2("Зіркова схема даних"),
                dcc.Graph(id="star-schema", figure=fig_star, style={"width": "1200px", "height": "800px"})
            ], style={"padding": "20px", "border": "2px solid #ccc", "margin-top": "20px"})
        ])
    ])
])


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
        .groupby("user_role").size().reset_index(name="кількість"),
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


if __name__ == '__main__':
    app.run(debug=True)
