import os, re, numpy as np, pandas as pd
from datetime import datetime
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.cluster import KMeans


class LevyAnalysis:
    """
    Клас для аналізу зборів та звітності.
    Завантажує дані з файлів:
      - levy.csv  (збори)
      - report.csv (звіти)
      - request.csv (запити, де "amount" у запиті – цільова сума)

    Розраховує:
      • Відсоток виконання зборів (ефективність = accumulated / target * 100)
      • Кількість звітів для кожного збору
      • Часовий тренд накопичення коштів із застосуванням Prophet
      • Матрицю кореляції між накопиченням, ефективністю та кількістю звітів
    """

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.levy_df = self.load_data("levy.csv")
        self.report_df = self.load_data("report.csv")
        self.request_df = self.load_data("request.csv")

    def load_data(self, filename):
        path = os.path.join(self.data_dir, filename)
        return pd.read_csv(path)

    def compute_levy_analysis(self):
        # Об'єднуємо дані зборів із запитами (де запит містить цільову суму)
        levy_req = pd.merge(self.levy_df, self.request_df, left_on="request_id", right_on="id",
                            suffixes=("_levy", "_req"))
        # Розрахунок ефективності зборів (у відсотках)
        levy_req["efficiency"] = (levy_req["accumulated"] / levy_req["amount"]) * 100
        # Розрахунок кількості звітів для кожного збору
        reports_count = self.report_df.groupby("levy_id").agg(report_count=("id", "count")).reset_index()
        self.levy_analysis = pd.merge(levy_req, reports_count, left_on="id_levy", right_on="levy_id", how="left")
        self.levy_analysis["report_count"] = self.levy_analysis["report_count"].fillna(0)
        return self.levy_analysis

    def time_trend_analysis(self):
        self.levy_analysis["create_date_levy"] = pd.to_datetime(self.levy_analysis["create_date_levy"], errors="coerce")
        trend_df = self.levy_analysis.groupby(self.levy_analysis["create_date_levy"].dt.date)[
            "accumulated"].sum().reset_index()
        trend_df.columns = ["ds", "y"]
        trend_df["ds"] = pd.to_datetime(trend_df["ds"])
        model = Prophet(daily_seasonality=True)
        model.fit(trend_df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        fig_trend = plot_plotly(model, forecast)
        fig_trend.update_layout(title="Часовий тренд зборів та прогноз накопичення коштів", height=600, width=1200)
        self.trend_figure = fig_trend
        return fig_trend

    def correlation_analysis(self):
        if not hasattr(self, "levy_analysis"):
            self.compute_levy_analysis()
        cols = ["accumulated", "efficiency", "report_count"]
        corr_matrix = self.levy_analysis[cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True,
                             title="Матриця кореляції: Накопичено, Ефективність, Кількість звітів",
                             labels={"color": "Коефіцієнт кореляції"})
        fig_corr.update_layout(height=600, width=1200)
        self.corr_figure = fig_corr
        return fig_corr

    def create_plots(self):
        if not hasattr(self, "levy_analysis"):
            self.compute_levy_analysis()
        # Альтернативна візуалізація: Scatter-графік ефективності vs. кількість звітів
        fig_scatter = px.scatter(self.levy_analysis, x="efficiency", y="report_count",
                                 size="accumulated", color="efficiency",
                                 hover_data=["id_levy", "accumulated"],
                                 title="Ефективність зборів vs. Кількість звітів")
        fig_scatter.update_layout(height=600, width=1200)
        fig_trend = self.time_trend_analysis()
        fig_corr = self.correlation_analysis()
        return fig_scatter, fig_trend, fig_corr

    def run(self):
        self.compute_levy_analysis()
        fig_scatter, fig_trend, fig_corr = self.create_plots()
        fig_scatter.write_html("levy_efficiency_scatter.html")
        fig_trend.write_html("levy_time_trend.html")
        fig_corr.write_html("levy_correlation_matrix.html")
        self.levy_analysis.to_csv("levy_analysis.csv", index=False)
        print("Аналіз зборів та звітності завершено.")
        print("Scatter-графік ефективності збережено у 'levy_efficiency_scatter.html'.")
        print("Графік часової серії з прогнозом збережено у 'levy_time_trend.html'.")
        print("Матриця кореляції збережена у 'levy_correlation_matrix.html'.")
