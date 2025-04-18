import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from sklearn.cluster import KMeans

# NLTK для аналізу сентименту та обробки тексту
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

nltk.download('vader_lexicon')
nltk.download('stopwords')
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))


class VolunteerAnalysis:
    """
    Клас для аналізу волонтерської діяльності та репутації.
    Завантажує дані з файлів volunteer.csv та volunteer_feedback.csv,
    проводить агрегування фідбеків, розрахунок сентименту, кластеризацію
    та тематичний аналіз текстів.
    """

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.volunteer_df = self.load_data("volunteer.csv")
        self.volunteer_feedback_df = self.load_data("volunteer_feedback.csv")

    def load_data(self, filename):
        path = os.path.join(self.data_dir, filename)
        return pd.read_csv(path)

    def compute_volunteer_analysis(self):
        """Обчислює агреговані показники для кожного волонтера та кластеризує дані."""
        feedback_agg = self.volunteer_feedback_df.groupby("volunteer_id").agg(
            average_reputation=("reputation_score", "mean"),
            feedback_count=("id", "count")
        ).reset_index()
        # Розрахунок сентименту
        self.volunteer_feedback_df["sentiment"] = self.volunteer_feedback_df["text"].apply(
            lambda x: sia.polarity_scores(x)["compound"]
        )
        sentiment_agg = self.volunteer_feedback_df.groupby("volunteer_id").agg(
            average_sentiment=("sentiment", "mean")
        ).reset_index()
        vol_analysis = pd.merge(feedback_agg, sentiment_agg, on="volunteer_id", how="left")
        # Кластеризація за допомогою KMeans
        features = vol_analysis[["average_reputation", "average_sentiment", "feedback_count"]].fillna(0)
        n_clusters = 3
        kmeans_vol = KMeans(n_clusters=n_clusters, random_state=42)
        vol_analysis["cluster"] = kmeans_vol.fit_predict(features)
        self.volunteer_analysis = vol_analysis
        return vol_analysis

    def thematic_analysis_feedback(self):
        """Проводить тематичний аналіз текстів фідбеків: повертає топ-10 ключових слів."""

        def extract_words(text):
            text = re.sub(r"[^\w\s]", "", text.lower())
            words = text.split()
            words = [w for w in words if w not in stop_words]
            return words

        all_words = self.volunteer_feedback_df["text"].apply(extract_words).explode()
        top_words = all_words.value_counts().head(10).reset_index()
        top_words.columns = ["слово", "кількість"]
        self.top_words = top_words
        return top_words

    def create_plots(self, cluster_chart_config=None, topwords_chart_config=None):
        """
        Створює графіки для аналізу волонтерів.
        :param cluster_chart_config: Словник для налаштування layout scatter-графіка
        :param topwords_chart_config: Словник для налаштування layout бар-графіка
        :return: (fig_scatter, fig_topwords)
        """
        if not hasattr(self, "volunteer_analysis"):
            self.compute_volunteer_analysis()
        if not hasattr(self, "top_words"):
            self.thematic_analysis_feedback()

        # Scatter-графік для кластеризації волонтерів
        fig_scatter = px.scatter(
            self.volunteer_analysis,
            x="average_reputation",
            y="average_sentiment",
            size="feedback_count",
            color="cluster",
            hover_data=["volunteer_id"],
            title="Кластеризація волонтерів: Репутація vs. Тональність"
        )
        if cluster_chart_config:
            fig_scatter.update_layout(**cluster_chart_config)
        else:
            fig_scatter.update_layout(height=600, width=1200)

        # Бар-графік для тематичного аналізу (вже встановлено орієнтацію горизонтально)
        fig_topwords = px.bar(
            self.top_words,
            x="кількість",
            y="слово",
            orientation="h",
            title="Топ-10 ключових слів у відгуках волонтерів"
        )
        if topwords_chart_config:
            # Видаляємо параметр 'orientation', якщо він присутній, щоб уникнути помилки
            tc = topwords_chart_config.copy()
            tc.pop("orientation", None)
            fig_topwords.update_layout(**tc)
        else:
            fig_topwords.update_layout(height=600, width=1200)

        return fig_scatter, fig_topwords

    def run(self):
        """Запускає аналіз і зберігає результати у файли."""
        self.compute_volunteer_analysis()
        self.thematic_analysis_feedback()
        fig_scatter, fig_topwords = self.create_plots()
        fig_scatter.write_html("volunteers_scatter.html")
        fig_topwords.write_html("top_words_volunteers.html")
        self.volunteer_analysis.to_csv("volunteer_analysis.csv", index=False)
        print("Аналіз волонтерської діяльності завершено.")
        print("Графік кластеризації збережено у 'volunteers_scatter.html'.")
        print("Графік топ-10 ключових слів збережено у 'top_words_volunteers.html'.")
        print("Таблиця з агрегованими даними збережена у 'volunteer_analysis.csv'.")
