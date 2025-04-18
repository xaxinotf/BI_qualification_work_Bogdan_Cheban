# retention_cohorts.py

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc

def get_cohort_controls(
    id_max_periods: str = "cohort-max-periods",
    id_min_users:   str = "cohort-min-users"
) -> html.Div:
    """
    Повертає блок елементів Dash для керування побудовою когортної матриці:
      - Ползунок для вибору максимальної кількості періодів (місяців)
      - Ползунок для мінімального розміру когорти
    """
    return html.Div(
        className="cohort-controls p-3 mb-4 border rounded bg-light",
        children=[
            html.Div([
                html.Label("Макс. місяців у когорті:", htmlFor=id_max_periods),
                dcc.Slider(
                    id=id_max_periods,
                    min=1, max=24, step=1, value=12,
                    marks={i: str(i) for i in range(1, 25, 3)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode="drag"
                )
            ], className="mb-4"),
            html.Div([
                html.Label("Мін. користувачів у когорті:", htmlFor=id_min_users),
                dcc.Slider(
                    id=id_min_users,
                    min=1, max=100, step=1, value=5,
                    marks={i: str(i) for i in range(1, 101, 10)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode="drag"
                )
            ])
        ]
    )

def make_cohort_figure(
    df: pd.DataFrame,
    max_periods:    int = 12,
    min_cohort_size:int = 5
) -> go.Figure:
    """
    Створює теплову карту Retention Cohorts на основі даних по донатам.

    Параметри:
    -----------
    df : pd.DataFrame
        DataFrame з колонками:
          - 'user_id'     : ідентифікатор користувача
          - 'create_date' : дата і час донату
    max_periods : int
        Максимальна кількість місяців для відображення по горизонталі.
    min_cohort_size : int
        Мінімальна кількість користувачів у когорті для відображення.

    Повертає:
    ---------
    go.Figure — об’єкт з тепловою картою.
    """
    # 1. Підготовка даних
    data = df.copy()
    data['order_date']   = pd.to_datetime(data['create_date'], errors='coerce')
    data = data.dropna(subset=['order_date'])
    data['order_period'] = data['order_date'].dt.to_period('M').dt.to_timestamp()
    data['cohort_period']= data.groupby('user_id')['order_period'].transform('min')

    data['period_number'] = (
        (data['order_period'].dt.year  - data['cohort_period'].dt.year) * 12 +
        (data['order_period'].dt.month - data['cohort_period'].dt.month)
    )
    # Обмежуємо по max_periods
    data = data[data['period_number'].between(0, max_periods)]

    # 2. Агреація
    cohorts = (
        data
        .groupby(['cohort_period', 'period_number'])['user_id']
        .nunique()
        .reset_index(name='n_users')
    )
    cohort_counts = (
        cohorts
        .pivot(index='cohort_period', columns='period_number', values='n_users')
        .fillna(0).astype(int)
    )

    # 3. Фільтр за мін. розміром
    cohort_sizes = cohort_counts.iloc[:, 0]
    valid_cohorts = cohort_sizes[cohort_sizes >= min_cohort_size].index
    cohort_counts = cohort_counts.loc[valid_cohorts]
    cohort_sizes  = cohort_sizes.loc[valid_cohorts]

    # 4. Обчислення відсотків утримання
    retention     = cohort_counts.div(cohort_sizes, axis=0)
    retention_pct = (retention * 100).round(1)

    # 5. Побудова Heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=retention_pct.values,
            x=[f"{i} міс." for i in retention_pct.columns],
            y=[d.strftime('%Y-%m') for d in retention_pct.index],
            text=retention_pct.values,
            texttemplate='%{text}%',
            colorscale='Blues',
            colorbar=dict(title='Утримання, %')
        )
    )
    fig.update_layout(
        title="Матриця утримання донорів (Retention Cohorts)",
        xaxis_title="Місяці від першого донату",
        yaxis_title="Когорта (місяць першого донату)",
        yaxis=dict(autorange='reversed'),
        template="plotly_white",
        margin=dict(l=60, r=30, t=80, b=60)
    )
    return fig

def make_cohort_size_figure(
    df: pd.DataFrame,
    min_cohort_size: int = 5
) -> go.Figure:
    """
    Створює стовпчикову діаграму розміру когорти
    (кількість унікальних донорів у кожному місяці-початку).
    """
    data = df.copy()
    data['order_date']   = pd.to_datetime(data['create_date'], errors='coerce')
    data = data.dropna(subset=['order_date'])
    data['order_period'] = data['order_date'].dt.to_period('M').dt.to_timestamp()
    data['cohort_period']= data.groupby('user_id')['order_period'].transform('min')

    cohort_sizes = (
        data
        .groupby('cohort_period')['user_id']
        .nunique()
        .sort_index()
    )
    cohort_sizes = cohort_sizes[cohort_sizes >= min_cohort_size]

    fig = px.bar(
        x=cohort_sizes.index.strftime('%Y-%m'),
        y=cohort_sizes.values,
        labels={'x': 'Когорта', 'y': 'Кількість користувачів'},
        title='Розмір когорти (початкові донори)',
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=60, r=30, t=80, b=60)
    )
    return fig
