# BI Qualification Work
проект реалізує інтерактивний BI Dashboard на основі Dash для аналізу донатів, заявок, волонтерської діяльності та зборів.


### **BI Application** — інтерактивна аналітична панель для аналізу донатів, заявок та поведінки користувачів. Поєднує класичну BI-звітність, прогнозування за допомогою Prophet, кластеризацію, OLAP-зіркову схему, retention-кохорти та модель прогнозування відтоку (churn).



### **Опис проєкту «Волонтер+»**

Платформа «Волонтер+» створена для швидкої та прозорої взаємодії між військовими, волонтерами та донорами. Основна програма BI Dashboard забезпечує:

•	Збір і агрегування даних із різних CSV-файлів системи («user.csv», «liqpay_order.csv», «request.csv» тощо), що дозволяє мати єдине сховище для подальшого аналізу.

•	Моніторинг фінансових потоків: аналіз загальних та середніх донатів, щоденні й місячні тренди, прогнозування обсягів допомоги за допомогою моделі Prophet.

•	Оцінку ефективності кампаній зборів («levy.csv» та «report.csv»): розрахунок коефіцієнту виконання, кількості звітів, часових трендів та кореляцій між показниками.

•	Сегментацію донорів і волонтерів: кластеризація за тривалістю перебування на платформі, сумами внесків або оцінками відгуків (K Means) для побудови таргетованих стратегій залучення.

•	Аналіз запитів і товарів: деталізація потреб військових бригад, агрегація за категоріями товарів, візуалізація у вигляді OLAP-зірки з усіма вимірами для швидкого drill down.

•	Retention Cohorts і Churn Prediction: візуалізація утримання донорів по місяцях та прогноз ризику відтоку для своєчасного реагування кореспондентів.

•	Інтерактивний гайд по інтерфейсу: модуль Spotlight-туру (guide.py), що допомагає новим користувачам швидко освоїтися з усіма можливостями Dashboard.


### Як програма допомагає аналізувати контент і дані

Завдяки єдиному інтерфейсу користувачі можуть:
1. Відстежувати в реальному часі ключові метрики (KPI) всіх волонтерських та донорських активностей.
2. Глибоко занурюватися в дані за допомогою інтерактивних фільтрів, clustering та drill‑down у зірковій схемі.
3. Приймати рішення на основі точного прогнозу потоків допомоги та аналізу утримання донорів.
4. Оцінювати якість роботи волонтерів через тематичний аналіз відгуків і репутаційні кластери.



### Демонстрація функціоналу програми
Нижче наведено приклади основних графіків і візуалізацій, які генерує програма

1.	Графік середнього внеску vs. тривалості перебування
![image](https://github.com/user-attachments/assets/58349f3a-9255-40d4-b18a-72e1e357803a)

2.	Тренд сукупних донатів по днях
![image](https://github.com/user-attachments/assets/86288939-97ab-4f19-9a1e-4c790772c5e6)

3.	Кореляційна матриця KPI
![image](https://github.com/user-attachments/assets/76d8a661-8605-41c2-925e-7dbe5068f9c4)
![image](https://github.com/user-attachments/assets/9e014a67-c7cd-41ac-8dd8-b15e80b43fd8)

4.	Зіркова схема OLAP‑куба
![image](https://github.com/user-attachments/assets/e2a962b1-0dcd-4b27-87b9-0329503db286)


5.	Сегментація донорів (K-Means)
![image](https://github.com/user-attachments/assets/137ac4bd-a100-4fa1-bee5-e4c8076e2c45)

6. Прогнозування внесків
![image](https://github.com/user-attachments/assets/4b4a0c25-56ef-497f-9fb6-51e4e06b6dc2)

7. Аналіз заявок за бригадами
![image](https://github.com/user-attachments/assets/db85e8eb-c59e-45cd-8f0d-f8a4684c4e46)

8. Аналіз заявок за товарами
![image](https://github.com/user-attachments/assets/9d10f903-9e65-449c-9ea6-f2a403ccc970)

9. Аналіз волонтерської діяльності та репутації
![image](https://github.com/user-attachments/assets/36819ef1-02e8-4356-a021-8810a2b743a1)

10.Аналіз зборів та звітності
![image](https://github.com/user-attachments/assets/8287555d-66de-4125-96eb-7e6ab18b9f14)

11.Утримання донорів (Retention Cohorts)
![image](https://github.com/user-attachments/assets/fe563528-5887-4da1-895f-b7beb8ce1090)

========================================================================================
![image](https://github.com/user-attachments/assets/fd2f9488-f26f-44c2-a25e-db7cb9213938)


## Структура проекту

```
├── bi.py                        # Основний запуск Dash-додатку
├── retention_cohorts.py         # Побудова когортних аналізів
├── guide.py                     # Інтерактивний Spotlight-тур
├── analysis_of_fees_and_reporting.py  # Клас LevyAnalysis
├── analysis_of_volunteer.py     # Клас VolunteerAnalysis
├── ai_check.py                  # Модуль для Churn Prediction
├── requirements.txt             # Перелік залежностей
├── data/                        # Папка з CSV-даними
│   ├── user.csv
│   ├── liqpay_order.csv
│   ├── request.csv
│   ├── military_personnel.csv
│   ├── brigade.csv
│   ├── levy.csv
│   ├── report.csv
│   └── ...                      # додаткові файли (email_template.csv, email_recipient.csv та ін.)
└── README.md
```

## Опис датасетів
===============================================================

user.csv – дані користувачів

volunteer.csv – дані волонтерів (підмножина користувачів)

volunteer_feedback.csv – відгуки про волонтерів

brigade.csv – дані про бригади

military_personnel.csv – дані про військовослужбовців

request.csv – запити на закупівлю (з використанням розширеного переліку товарів)

levy.csv – дані про збори (накопичення та трофеї)

volunteer_levy.csv – зв’язок між волонтерами та зборами

report.csv – звіти, пов’язані із зборами

attachment.csv – файли-атачменти до звітів

brigade_codes.csv – коди бригад

add_request.csv – дані для додавання військового (незареєстрованого)

email_template.csv – шаблони електронних листів

email_notification.csv – сповіщення за шаблонами

email_recipient.csv – інформація про отримувачів email‑сповіщень

email_attachment.csv – додатки до email‑сповіщень

liqpay_order.csv – транзакції (донати)

ai_chat_messages.csv – історія повідомлень для AI‑чату

===============================================================

Нижче наведено детальний опис основних файлів даних, їхніх стовпців та ролі у Dashboard:

* **user.csv**

  * `id`: унікальний ідентифікатор користувача.
  * `create_date`: дата та час реєстрації на платформі.
  * `user_role`: роль користувача (наприклад, donor, volunteer, admin).
    Використовується для аналізу життєвого циклу користувача, тривалості перебування, побудови когорт.

* **liqpay\_order.csv**

  * `id`: унікальний ідентифікатор транзакції.
  * `user_id`: посилання на `id` користувача з `user.csv`.
  * `create_date`: дата й час проведення платежу.
  * `amount`: сума донату.
    Використовується для розрахунку загальних та середніх донатів, трендів і прогнозів.

* **request.csv**

  * `id`: унікальний ідентифікатор заявки.
  * `user_id`: посилання на `id` користувача, що створив заявку.
  * `military_personnel_id`: посилання на `id` військовослужбовця з `military_personnel.csv`.
  * `brigade_id`: посилання на `id` бригади з `brigade.csv`.
  * `amount`: запитувана сума.
  * `description`: текстовий опис запиту із зазначенням товару або послуги.
    Використовується для аналізу потреб за бригадами та категоріями товарів.

* **military\_personnel.csv**

  * `id`: унікальний ідентифікатор військовослужбовця.
  * `name`: ім’я та прізвище військовослужбовця.
  * Додаткові стовпці з деталями (звання, підрозділ тощо).
    Дозволяє зв’язати заявки з конкретними військовими.

* **brigade.csv**

  * `id`: унікальний ідентифікатор бригади.
  * `name`: назва або шифр бригади.
    Використовується для агрегації заявок та сум за підрозділами.

* **levy.csv**

  * `id`: унікальний ідентифікатор кампанії збору коштів.
  * `request_id`: посилання на `id` заявки, що ініціювала збір.
  * `accumulated`: сума зібраних коштів станом на дату.
  * `create_date_levy`: дата та час запису стану збору.
    Використовується для аналізу ефективності, трендів та прогнозування кампаній.

* **report.csv**

  * `id`: унікальний ідентифікатор звіту.
  * `levy_id`: посилання на `id` збору з `levy.csv`.
  * Додаткові поля: `report_date` (дата звіту), `details` (опис звіту).
    Служить для підрахунку кількості звітів та оцінки прозорості кампаній.

* **volunteer.csv**

  * `id`: унікальний ідентифікатор запису.
  * `user_id`: посилання на волонтера в `user.csv`.
  * `feedback_score`: числова оцінка відгуку.
  * `feedback_text`: текстовий відгук.
  * `activity_date`: дата активності.
    Використовується для аналізу репутації волонтерів та тематичного аналізу відгуків.

* **volunteer\_levy.csv**

  * `id`: унікальний ідентифікатор зв’язку.
  * `volunteer_id`: посилання на `id` волонтера.
  * `levy_id`: посилання на `id` кампанії збору.
    Дозволяє аналізувати внесок волонтерів у різні кампанії.

* **email\_template.csv**

  * `template_id`: унікальний ідентифікатор шаблону листа.
  * `subject`: тема листа.
  * `body`: тіло листа (HTML/текст).
    Використовується у зірковій схемі як вимір «Email Template».

* **email\_recipient.csv**, **email\_notification.csv**, **email\_attachment.csv**
  Додаткові таблиці для OLAP-зірки: зв’язки між шаблонами, одержувачами та вкладеннями.

* **ai\_chat\_messages.csv**

  * `message_id`: унікальний ідентифікатор повідомлення чат-бота.
  * `user_id`: посилання на користувача.
  * `message_text`: текст повідомлення.
  * `timestamp`: дата та час надсилання.
    Додано до зіркової схеми як приклад джерела даних AI.

Цей розгорнутий опис допоможе швидко зорієнтуватися в структурі даних і їхній ролі у Dashboard.

## Встановлення та запуск

1. Клонувати репозиторій:

   ```bash
   git clone https://github.com/xaxinotf/BI_qualification_work_Bogdan_Cheban.git
   cd BI_qualification_work_Bogdan_Cheban
   ```
2. Встановити залежності:

   ```bash
   pip install pandas, numpy, plotly, prophet, scikit-learn, dash, dash-bootstrap-components, dash-cytoscape, dask, pyspark
   ```
3. Запустити додаток:

   ```bash
   python bi.py
   ```
4. Відкрити в браузері: `http://127.0.0.1:8050`

## Залежності

* **Python 3.8+** — рекомендована мінімальна версія інтерпретатора для сумісності з усіма використовуваними бібліотеками та модулями.
* **pandas** — потужна бібліотека для обробки та аналізу табличних даних (DataFrame), підтримує злиття, агрегації, очистку та перетворення даних.
* **numpy** — базова бібліотека для числових обчислень, багатовимірних масивів та високопродуктивних математичних операцій.
* **plotly** — фреймворк для створення інтерактивних веб‑графіків (лінійні графіки, гістограми, теплові карти та ін.), що легко інтегрується з Dash.
* **prophet** — бібліотека від Meta для автоматизованого моделювання часових рядів з підтримкою сезонності, трендів та свят, використовується для прогнозування обсягів донатів і зборів.
* **scikit-learn** — набір інструментів для машинного навчання (кластеризація, класифікація, регресія), в нашому випадку застосовується алгоритм K‑Means для сегментації донорів.
* **dash** — фреймворк для створення інтерактивних аналітичних веб‑додатків на Python без необхідності писати JavaScript.
* **dash-bootstrap-components** — набір готових компонентів Bootstrap для Dash, що дозволяє швидко стилізувати інтерфейс.
* **dash-cytoscape** — компонент для візуалізації графів і мереж (наприклад, зіркова схема OLAP) у Dash.
* **dask** (опційно) — бібліотека для масштабованої обробки великих обсягів даних з API, сумісним з pandas, забезпечує розподілені обчислення.
* **pyspark** (опційно) — інтеграція з Apache Spark для роботи з дуже великими наборами даних у кластерних середовищах.

## Використання

Інтерфейс Dashboard побудований за модульним принципом з лівого бокового меню, що дозволяє швидко переключатися між ключовими розділами аналітики:

* **Аналітика пожертв та збір коштів**: миттєве відображення загальних, середніх та динамічних показників транзакцій, глибокий аналіз трендів та прогнозування обсягів донатів за допомогою Prophet.
* **Сегментація донорів та волонтерів**: застосування алгоритму K‑Means для виявлення груп користувачів за тривалістю перебування та сумами внесків або оцінками відгуків, що допомагає формувати таргетовані кампанії.
* **Retention Cohorts та Churn Prediction**: аналіз утримання донорів по когортах із подальшим передбаченням відтоку для своєчасного реагування та побудови стратегій утримання.
* **Аналіз заявок та товарів**: деталізовані метрики за військовими бригадами з фільтрацією по типу запиту та категоріях товарів, розгортання OLAP-зірки для комплексного перегляду зв’язків між сутностями платформи.
* **Глибокий аналіз волонтерського контенту**: кластеризація за репутацією, тематичний аналіз відгуків з виділенням ключових слів, що дає змогу оцінити якість взаємодій і покращити комунікацію.

Кожен модуль оснащений інтерактивними фільтрами (дропдауни, слайдери, пошук) та візуалізацією (графіки Plotly), що дозволяє аналітикам та менеджерам отримувати відповіді на запити «що відбувається, чому та що робити далі» за лічені секунди.

## Перспективи розширення

--**Розширення джерел даних** — інтеграція зовнішніх API (наприклад, соціальні мережі, CRM-системи, GPS‑треки доставки) для більш глибинного аналізу ланцюгів постачання.

--**Розвиток AI/ML** — впровадження автоматичного класифікатора запитів, рекомендаторів волонтерів, продвинутої моделей прогнозування та обробки природної мови для аналізу вільного тексту запитів і відгуків.

--**Реал‑тайм дашборди** — перехід від пакетного оновлення даних до стрімінг-обробки (Kafka, Spark Streaming) для моніторингу в режимі реального часу.

--**Мобільні клієнти** — адаптація інтерфейсу для мобільних пристроїв та випуск нативних додатків для волонтерів і військових.

--**Побудова механізму зворотного зв’язку** — інтеграція систем оцінки задоволеності, опитувань та сповіщень для підвищення залученості та прозорості процесу.

## Ліцензія та юридична інформація

![volonteerr+_mit_licence](https://github.com/user-attachments/assets/48e5a528-79da-4cf0-b3a9-ede18354f562)


Цей проєкт «Волонтер+» розповсюджується на умовах ліцензії MIT. Ви можете вільно використовувати, копіювати, модифікувати та поширювати код за умови збереження авторських прав та вказання джерела.


## Відповідність законодавству

**Платформа "Волонтер+" відповідає чинному законодавству України, зокрема:**

**Закон України «Про захист персональних даних» №2297-VI – забезпечення захисту та конфіденційності персональної інформації користувачів.**

**Закон України «Про волонтерську діяльність» №3236-VI – регулювання відносин у сфері волонтерської діяльності, створення умов для її розвитку.**

**Закон України «Про інформацію» №2657-XII – регулювання доступу до інформації, її обробки, поширення та захисту.**

**Закон України «Про електронні довірчі послуги» №2155-VIII – забезпечення юридичної значущості електронних транзакцій і документів, які обробляються платформою.**

## Захист даних

-- Усі дані, що обробляються в рамках платформи "Волонтер+", надійно захищені. Використовуються сучасні технології шифрування (SSL/TLS), багаторівнева автентифікація користувачів, регулярні бекапи та політики контролю доступу, що дозволяє запобігти несанкціонованому доступу, витоку інформації або її втраті. Платформа постійно проходить аудит на відповідність стандартам інформаційної безпеки.

-- Використовуючи платформу, ви погоджуєтесь із наведеними умовами ліцензії та політикою конфіденційності.

### Перспективи розширення

* Інтеграція з зовнішніми API (CRM, ERP, логістичні сервіси) для поповнення бази даних та підвищення глибини аналітики.
* Додати систему рекомендацій волонтерських профілів та автоматичну обробку текстових запитів за допомогою NLP.
* Реалізація режиму реального часу з потоковою обробкою даних (Kafka, Spark Streaming) для миттєвого оновлення показників.
* Розширення мобільного інтерфейсу та створення нативних додатків для волонтерів і військових.

