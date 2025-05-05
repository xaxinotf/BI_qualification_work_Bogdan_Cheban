import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback_context

"""
Модуль гіда (spot‑light тур) — покрокова інтерактивна екскурсія по інтерфейсу.
Використовує CSS-змінні дизайн‑системи з Sass‑фолбеками; адаптивні popover;
покращена доступність і чітке розділення кроків.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Стилі та допоміжні функції
# ──────────────────────────────────────────────────────────────────────────────

OLIVE_FALLBACK       = "#556b2f"  # відповідник дизайн olive-500
OLIVE_LIGHT_FALLBACK = "#829661"  # відповідник дизайн olive-300

GUIDE_STYLE = dcc.Markdown(
    f"""
<style>
:root {{
  --g-primary:       var(--color-olive-500, {OLIVE_FALLBACK});
  --g-primary-light: var(--color-olive-300, {OLIVE_LIGHT_FALLBACK});
  --g-bg:            var(--color-gray-50,  #f1f1f1);
  --g-shadow:        rgba(0,0,0,0.35);
}}

/* Тло під час туру */
.guide-active::before {{
  content: ''; position: fixed; inset: 0;
  background: var(--g-shadow);
  backdrop-filter: blur(3px);
  z-index: 999;
}}

/* Підсвічена ціль */
.guide-highlight {{
  position: relative !important; z-index: 1000 !important;
  box-shadow: 0 0 0 6px var(--g-primary)88, 0 0 30px var(--g-primary)BB;
  border-radius: 8px;
  transition: transform .3s, box-shadow .3s;
}}
.guide-highlight:hover {{ transform: scale(1.03); }}

/* Вікно з описом кроку */
.popover {{
  max-width: 500px;
  border: 2px solid var(--g-primary);
  border-radius: 1rem;
  background: var(--color-white, #fff);
  box-shadow: 0 16px 48px rgba(0,0,0,0.2);
  animation: fadeSlide 0.4s ease both;
  font-family: 'Montserrat', sans-serif;
}}
.popover-header {{
  background: var(--g-primary);
  color: var(--color-white, #fff);
  font-size: 1.1rem;
  font-weight: 600;
}}
.popover-body {{
  font-size: 1rem;
  line-height: 1.6;
  color: var(--color-gray-900, #171717);
}}
@keyframes fadeSlide {{
  from {{ opacity: 0; transform: translateY(-12px); }}
  to   {{ opacity: 1; transform: none; }}
}}

/* Кнопка запуску туру */
#guide-start {{
  width: 3.5rem; height: 3.5rem;
  font-size: 1.75rem;
  border-radius: 50%;
  background: var(--g-primary);
  color: var(--color-white, #fff);
  box-shadow: 0 6px 16px rgba(0,0,0,0.3);
  transition: transform .2s;
}}
#guide-start:hover {{ transform: scale(1.15); }}

/* Кнопки навігації в popover */
.guide-ctrl {{ min-width: 6rem; font-size: .85rem; }}
</style>
""",
    dangerously_allow_html=True,
    style={"display": "none"},
)


def S(title: str, body: str, target: str, placement: str = "auto") -> dict:
    """Створює словник кроку туру з заголовком, описом, ціллю та позиціюванням."""
    return {"title": title, "body": body, "target": target, "placement": placement}

# ──────────────────────────────────────────────────────────────────────────────
# Кроки туру: спочатку кожна вкладка меню, потім детальні елементи
# ──────────────────────────────────────────────────────────────────────────────

tab_steps = [
    ("📊 Аналітика",          "Ключові графіки, кореляції, KPI та швидкі фільтри.",                   "sidebar-tabs", "right"),
    ("🌟 Зіркова схема",       "Інтерактивна OLAP‑модель (факт/виміри) з можливістю drill-down.",        "sidebar-tabs", "right"),
    ("📊 Сегментація донорів", "K‑Means кластери донорів: суми, тривалість, профілі груп.",              "sidebar-tabs", "right"),
    ("🔮 Прогнозування",       "Prophet‑модель: налаштуйте горизонт та подивіться майбутні тенденції.",  "sidebar-tabs", "right"),
    ("🎯 Керування KPI",       "Встановлюйте цільові показники й порівнюйте з фактичними даними.",      "sidebar-tabs", "right"),
    ("📝 Запити",              "Метрики заявок по військових бригадах із вибором показників.",         "sidebar-tabs", "right"),
    ("📦 Аналіз товарів",       "Популярність і суми заявок за номенклатурою товарів.",                 "sidebar-tabs", "right"),
    ("🤝 Волонтерський аналіз", "Репутаційні кластери та топ‑слова у волонтерських відгуках.",      "sidebar-tabs", "right"),
    ("💰 Аналіз зборів",        "Ефективність кампаній та кореляції між метриками зборів.",            "sidebar-tabs", "right"),
    ("📆 Retention Cohorts",    "Утримання донорів у часі: налаштуйте горизонти когорт та мін. розмір.",  "sidebar-tabs", "right"),
    ("⚠️ Churn Prediction",     "Прогноз відтоку: ROC‑модель та рекомендації для ризикових донорів.",    "sidebar-tabs", "right"),
]

detail_steps = [
    ("🎛 Фільтри аналітики",    "Фільтруйте роль та тривалість перебування для точного аналізу.",           "role-filter", "right"),
    ("📈 Середній донат vs час", "Графік: дні на платформі (X) vs середній донат (Y), колір — роль.", "scatter-graph", "bottom"),
    ("🌐 OLAP вузол",           "Клацніть вузол на зірковій схемі для деталізації факт-таблиці.",         "star-schema", "bottom"),
    ("🔢 Фільтр кластерів",     "Оберіть кластери на дашборді сегментації донорів.",                  "cluster-filter", "right"),
    ("📅 Горизонт прогнозу",     "Перетягніть слайдер, щоб змінити кількість днів у прогнозі.",           "forecast-horizon-slider", "bottom"),
    ("⚙️ Оновити KPI",          "Після введення нових цілей натисніть кнопку для оновлення статусу.",  "update-kpi-btn", "left"),
    ("📊 Метрика заявок",        "Виберіть: кількість, загальна сума або середня сума заявки.",        "request-metric-dropdown", "right"),
    ("🔍 Фільтр товарів",        "Шукайте за ключовим словом у темі запиту.",                      "product-topic-filter", "right"),
    ("🤖 Кластер волонтерів",    "Наведіть на точку, щоб побачити репутацію та тональність.",     "volunteer-scatter", "bottom"),
    ("📈 Тренд зборів",          "Тренд по сумах зборів з часом.",                                 "levy-scatter", "bottom"),
    ("📊 Cohort controls",       "Налаштуйте max когорти та мінімум користувачів.",                  "cohort-max-periods", "right"),
    ("📉 ROC крива",             "Оцініть якість моделі відтоку.",                                 "churn-roc-curve", "left"),
]

GUIDE_STEPS = [*map(lambda x: S(*x), tab_steps), *map(lambda x: S(*x), detail_steps)]

# ──────────────────────────────────────────────────────────────────────────────
# Макет та колбеки (без змін)
# ──────────────────────────────────────────────────────────────────────────────

GUIDE_LAYOUT = html.Div([
    GUIDE_STYLE,
    dcc.Store(id="guide-step", data=-1),
    dbc.Button("❓", id="guide-start", color="success", className="position-fixed top-0 end-0 m-3 zindex-tooltip", title="Швидка екскурсія"),
    dbc.Popover(
        id="guide-popover",
        is_open=False,
        target="guide-dummy",
        placement="auto",
        hide_arrow=False,
        class_name="popover",
        children=[
            dbc.PopoverHeader(id="guide-popover-title", className="fw-bold"),
            dbc.PopoverBody(id="guide-popover-body"),
            html.Div([
                dbc.Button("← Назад", id="guide-back", size="sm", className="guide-ctrl me-2 btn-outline-secondary"),
                dbc.Button("Далі →", id="guide-next", size="sm", color="primary", className="guide-ctrl"),
                dbc.Button("× Завершити", id="guide-finish", size="sm", color="danger", className="ms-2 guide-ctrl"),
            ], className="d-flex justify-content-end mt-3"),
        ],
    ),
    html.Div(id="guide-dummy", style={"display": "none"}),
])


def register_guide(app: dash.Dash):
    # Навігація між кроками туру
    @app.callback(
        Output("guide-popover", "is_open"), Output("guide-popover", "target"), Output("guide-popover", "placement"),
        Output("guide-popover-title", "children"), Output("guide-popover-body", "children"), Output("guide-step", "data"),
        Output("guide-back", "disabled"), Output("guide-next", "disabled"), Output("guide-finish", "style"),
        Input("guide-start", "n_clicks"), Input("guide-next", "n_clicks"), Input("guide-back", "n_clicks"), Input("guide-finish", "n_clicks"),
        State("guide-step", "data"), prevent_initial_call=True
    )
    def _nav(start, nxt, back, fin, step):
        btn = callback_context.triggered[0]["prop_id"].split('.')[0]
        if btn == "guide-start":
            step = 0
        elif btn == "guide-next":
            step += 1
        elif btn == "guide-back":
            step = max(step - 1, 0)
        elif btn == "guide-finish":
            return (False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, -1, True, True, {"display": "none"})
        if step < 0 or step >= len(GUIDE_STEPS):
            return (False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, -1, True, True, {"display": "none"})
        info = GUIDE_STEPS[step]
        return (
            True,
            info['target'],
            info.get('placement','auto'),
            info['title'],
            info['body'],
            step,
            step == 0,
            step == len(GUIDE_STEPS) - 1,
            {"display": "inline-block"} if step == len(GUIDE_STEPS) - 1 else {"display": "none"}
        )

    # Підсвічування поточної цілі
    for tid in {s['target'] for s in GUIDE_STEPS}:
        def mk(t=tid):
            @app.callback(Output(t, 'className', allow_duplicate=True), Input('guide-step','data'), State(t,'className'), prevent_initial_call=True)
            def hl(step, cls):
                classes = [c for c in (cls or '').split() if c!='guide-highlight']
                if 0<=step<len(GUIDE_STEPS) and GUIDE_STEPS[step]['target']==t: classes.append('guide-highlight')
                return ' '.join(classes)
        mk()

    # Додавання/зняття overlay класу на <body>
    app.clientside_callback(
        "function(o){document.body.classList.toggle('guide-active',o);return ''}",
        Output('guide-dummy','children'), Input('guide-popover','is_open')
    )
