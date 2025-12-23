import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re

st.set_page_config(
    page_title="Предсказание цен на автомобили",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ЗАГРУЗКА ДАННЫХ И МОДЕЛЕЙ
# ============================================================================

@st.cache_data
def load_model_and_objects():
    """Загружает модель и вспомогательные объекты"""
    files = {
        'pipeline': 'model_pipeline.pkl',
        'ohe_encoder': 'ohe_encoder.pkl',
        'feature_info': 'feature_info.pkl'
    }
    loaded = {}
    try:
        for name, path in files.items():
            with open(path, 'rb') as f:
                loaded[name] = pickle.load(f)
        return loaded['pipeline'], loaded['ohe_encoder'], loaded['feature_info']
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки файла: {e}")
        st.info("Убедитесь, что файлы .pkl находятся в рабочей директории.")
        return None, None, None

@st.cache_data
def load_eda_data():
    try:
        return pd.read_csv('train_data_for_eda.csv')
    except FileNotFoundError:
        return None

pipeline, ohe_encoder, feature_info = load_model_and_objects()
df_eda = load_eda_data()

# ============================================================================
# ФУНКЦИИ ОБРАБОТКИ ДАННЫХ
# ============================================================================

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует инженерные признаки из сырых данных.
    Использует векторизованные операции для скорости.
    """
    df = df.copy()
    
    if 'name' in df.columns:
        df['name_clean'] = df['name'].astype(str).str.lower().fillna('')
        
        if 'brand' not in df.columns:
            df['brand'] = df['name_clean'].str.split().str[0]
        if 'model' not in df.columns:
            df['model'] = df['name'].str.split().str[1].fillna('Unknown')

        df['name_len'] = df['name'].astype(str).str.len().fillna(0)
        
        keywords = {
            'is_luxury': r'bmw|mercedes|audi|jaguar|porsche|lexus|volvo|land',
            'has_turbo': r'turbo',
            'is_special_edition': r'special|edition|limited',
            'has_plus_pro': r'plus|pro|premium',
            'is_base_trim': r'base|std|lx|lxi|e\b',
            'is_old_bs': r'bs3|bs4|bsiii|bsiv',
            'is_4wd': r'4wd|4x4|awd',
            'diesel_in_name': r'diesel'
        }

        for col, pattern in keywords.items():
            if col not in df.columns:
                df[col] = df['name_clean'].str.contains(pattern, regex=True).astype(int)
        
        df.drop(columns=['name_clean'], inplace=True, errors='ignore')
    
    else:
        cols_to_fill = ['name_len', 'is_luxury', 'has_turbo', 'is_special_edition', 
                        'has_plus_pro', 'is_base_trim', 'is_old_bs', 'is_4wd', 'diesel_in_name']
        for col in cols_to_fill:
            if col not in df.columns:
                df[col] = 0

    return df

def prepare_input_for_model(df_raw, encoder, f_info):
    """
    Полный цикл подготовки: препроцессинг -> OHE -> выравнивание колонок
    """
    df_processed = preprocess_features(df_raw)
    
    num_cols = [c for c in f_info['numeric_features'] if c in df_processed.columns]
    cat_cols = [c for c in f_info['categorical_features'] if c in df_processed.columns]
    
    numeric_data = df_processed[num_cols].reset_index(drop=True)
    categorical_data = df_processed[cat_cols].reset_index(drop=True)
    
    try:
        cat_encoded = encoder.transform(categorical_data)
        cat_encoded_df = pd.DataFrame(
            cat_encoded,
            columns=encoder.get_feature_names_out(categorical_data.columns)
        )
    except Exception as e:
        st.error(f"Ошибка кодирования категорий. Проверьте входные данные. {e}")
        return None

    input_data = pd.concat([numeric_data, cat_encoded_df], axis=1)
    
    final_cols = f_info['feature_names_after_ohe']
    input_data = input_data.reindex(columns=final_cols, fill_value=0)
    
    return input_data

# ============================================================================
# UI ФУНКЦИИ (СТРАНИЦЫ)
# ============================================================================

def show_eda_page():
    st.header("Exploratory Data Analysis (EDA)")
    
    if df_eda is None:
        st.warning("Данные train_data_for_eda.csv не найдены.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Всего записей", len(df_eda))
    col2.metric("Признаков", df_eda.shape[1])
    col3.metric("Целевая переменная", "selling_price")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Распределение цен", "Год выпуска", "Корреляции"])
    
    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(df_eda['selling_price'], bins=50, ax=axes[0], kde=True)
        axes[0].set_title('Цена продажи')
        
        sns.histplot(np.log1p(df_eda['selling_price']), bins=50, ax=axes[1], color='orange', kde=True)
        axes[1].set_title('Log(Цена продажи)')
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df_eda, x='year', y='selling_price', alpha=0.4, size='km_driven', sizes=(10, 200))
        ax.set_title('Цена vs Год выпуска (размер точки = пробег)')
        st.pyplot(fig)

    with tab3:
        numeric_cols = df_eda.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(df_eda[numeric_cols].corr(), dtype=bool))
            sns.heatmap(df_eda[numeric_cols].corr(), mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, ax=ax)
            st.pyplot(fig)

def show_prediction_page():
    st.header("Предсказание цены")
    
    if not all([pipeline, ohe_encoder, feature_info]):
        st.error("Необходимые компоненты модели не загружены.")
        return

    tab_manual, tab_csv = st.tabs(["Ручной ввод", "Загрузка файла"])

    with tab_manual:
        with st.form("manual_input_form"):
            col1, col2 = st.columns(2)
            with col1:
                year = st.number_input("Год", 1900, 2024, 2015)
                km_driven = st.number_input("Пробег (км)", 0, None, 50000)
                mileage = st.number_input("Расход (kmpl)", 0.0, None, 20.0)
                engine = st.number_input("Объем (CC)", 0, None, 1200)
                max_power = st.number_input("Мощность (bhp)", 0.0, None, 80.0)
            
            with col2:
                torque = st.number_input("Момент (Nm)", 0.0, None, 150.0)
                max_torque_rpm = st.number_input("Обороты макс. момента", 0, None, 3000)
                fuel = st.selectbox("Топливо", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
                seller_type = st.selectbox("Продавец", ["Individual", "Dealer", "Trustmark Dealer"])
                transmission = st.selectbox("КПП", ["Manual", "Automatic"])
                owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
                seats = st.number_input("Мест", 2, 14, 5)
            
            name = st.text_input("Полное название авто", "Maruti Swift Dzire VDI")
            
            submitted = st.form_submit_button("Рассчитать цену", type="primary")

            if submitted:
                raw_data = pd.DataFrame({
                    'year': [year], 'km_driven': [km_driven], 'mileage': [mileage],
                    'engine': [engine], 'max_power': [max_power], 'torque': [torque],
                    'max_torque_rpm': [max_torque_rpm], 'fuel': [fuel],
                    'seller_type': [seller_type], 'transmission': [transmission],
                    'owner': [owner], 'seats': [seats], 'name': [name]
                })

                X = prepare_input_for_model(raw_data, ohe_encoder, feature_info)
                if X is not None:
                    pred = pipeline.predict(X)[0]
                    st.success(f"###Оценочная стоимость: {pred:,.0f} рупий")

    with tab_csv:
        uploaded_file = st.file_uploader("Загрузите CSV с параметрами авто", type=['csv'])
        if uploaded_file:
            df_input = pd.read_csv(uploaded_file)
            st.dataframe(df_input, use_container_width=True)

            if st.button("Предсказать для всех"):
                with st.spinner("Обработка данных..."):
                    if 'name' not in df_input.columns and 'brand' not in df_input.columns:
                        st.error("В файле должен быть столбец 'name' или 'brand'!")
                    else:
                        X = prepare_input_for_model(df_input, ohe_encoder, feature_info)
                        
                        if X is not None:
                            predictions = pipeline.predict(X)
                            df_result = df_input.copy()
                            df_result['Predicted_Price'] = predictions
                            
                            st.success("Готово!")
                            st.dataframe(df_result)
                            
                            csv = df_result.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Скачать результат", 
                                csv, "predictions.csv", "text/csv", 
                                key='download-csv'
                            )

def show_weights_page():
    st.header("Интерпретация модели")
    model_step = pipeline.named_steps.get('model', pipeline) 
    coefs = model_step.coef_
    features = feature_info['feature_names_after_ohe']
    
    df_weights = pd.DataFrame({'Feature': features, 'Weight': coefs})
    df_weights['Abs_Weight'] = df_weights['Weight'].abs()
    df_weights = df_weights.sort_values('Weight', ascending=False).reset_index(drop=True)

    st.subheader("Топ-15 Влиятельных признаков")
    
    top_n = df_weights.head(15)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#ff4b4b' if x < 0 else '#4caf50' for x in top_n['Weight']]
    sns.barplot(data=top_n, x='Weight', y='Feature', palette=colors, ax=ax)
    ax.set_title("Веса признаков (Зеленый +, Красный -)")
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)

    with st.expander("Посмотреть все веса"):
        st.dataframe(df_weights[['Feature', 'Weight']], use_container_width=True)

# ============================================================================
# ГЛАВНОЕ МЕНЮ
# ============================================================================

st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти к:", ["Анализ данных", "Предсказание", "Веса модели"])

if page == "Анализ данных":
    show_eda_page()
elif page == "Предсказание":
    show_prediction_page()
elif page == "Веса модели":
    show_weights_page()

st.sidebar.markdown("---")
st.sidebar.info("Алябьев Р.Р | Streamlit")