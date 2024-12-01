import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Configuración de la aplicación Streamlit
st.set_page_config(layout="wide", page_title="Modelo de Predicción", page_icon="🔮")
st.title("🔜 Modelo de Predicción con Random Forest")
st.write("Este modelo utiliza las mejores variables predictoras seleccionadas para predecir la cantidad comprada 🛒")

# Cargar datos
uploaded_file = "D:/Documentos/VALERIA/Documents/Univalle 4 Vale/Proyecto Integrador/PilAndina/PilAndina/df/optimized_dataset.xlsx"
data = pd.read_excel(uploaded_file)
st.write("Datos cargados exitosamente:")
st.dataframe(data)

# Codificar la variable 'nombre_categoria' con LabelEncoder
if 'nombre_categoria' in data.columns:
    label_encoder_categoria = LabelEncoder()
    data['nombre_categoria_encoded'] = label_encoder_categoria.fit_transform(data['nombre_categoria'])
    st.write("Variable 'nombre_categoria' codificada exitosamente.")
else:
    st.error("La columna 'nombre_categoria' no se encuentra en el dataset.")

# Seleccionar las columnas relevantes
predictors = ['precio', 'vida_util', 'precio_competencia', 'año', 'historial_compras', 'cantidad_disponible', 'nombre_categoria_encoded']
target = 'cantidad_comprada'

# Validar que las columnas existan en el archivo cargado
missing_columns = [col for col in predictors + [target] if col not in data.columns]
if missing_columns:
    st.error(f"Las siguientes columnas faltan en los datos cargados: {missing_columns}")
else:
    # Separar las variables predictoras y dependiente
    X = data[predictors]
    y = data[target]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo de Random Forest
    model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=20)
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas de rendimiento
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Mostrar resultados
    st.write("### Resultados del Modelo")
    st.write(f"- RMSE (Raíz del error cuadrático medio): {rmse:.2f}")
    st.write(f"- MAE (Error absoluto medio): {mae:.2f}")
    st.write(f"- R² (Coeficiente de determinación): {r2:.2f}")

    # Importancia de las características
    feature_importance = pd.DataFrame({
    "Variable": predictors,
    "Importancia": model.feature_importances_
    }).sort_values(by="Importancia", ascending=False)

    # Colores específicos
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FF8C33', '#A133FF', '#33FFF6']

    # Gráfica
    fig = px.bar(
        feature_importance,
        x="Variable",
        y="Importancia",
        color="Variable",
        color_discrete_sequence=colors,
        title="Importancia de las Variables Predictoras"
    )

    fig.update_layout(
    xaxis=dict(tickangle=45),  # Rotar etiquetas del eje X
    title_x=0.5               # Centrar el título
    )

    # Mostrar la gráfica directamente en la página de Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    # Predicción de ventas hasta 2026
    st.write("### Predicción de cantidad de ventas hasta 2023")
    
    # Filtrar solo columnas numéricas
    numeric_columns = data.select_dtypes(include=['number']).columns

    # Agrupar por 'año' y calcular la media para columnas numéricas
    future_years = list(range(2019, 2024))
    future_data = data.groupby('año')[numeric_columns].mean().reindex(future_years)

    # Reiniciar el índice sin duplicar la columna 'año'
    if 'año' in future_data.columns:
        future_data = future_data.rename(columns={'año': 'año_existente'})  # Renombrar temporalmente para evitar duplicados
    future_data = future_data.reset_index()  # Ahora 'año' será el nuevo índice

    # Ajustar los datos para las predicciones futuras
    future_data[predictors] = future_data[predictors].fillna(method='ffill')
    future_data['cantidad_comprada'] = model.predict(future_data[predictors])

    # Gráfico de línea de tendencia
    plt.figure(figsize=(10, 6))
    plt.plot(future_data['año'], future_data['cantidad_comprada'], marker='o', linestyle='-', label="Predicción de ventas")
    plt.title("Cantidad de Ventas (2019-2023)")
    plt.xlabel("Año")
    plt.ylabel("Cantidad Comprada")
    plt.grid()
    plt.legend()
    st.pyplot(plt)

    # Predicción personalizada
    st.write("## Predicción personalizada")

    # Entradas personalizadas para la predicción
    categoria_seleccionada = st.selectbox("Selecciona la categoría", label_encoder_categoria.classes_)
    precio = st.number_input("Precio", min_value=0.0, value=10.0)
    vida_util = st.number_input("Vida útil (días)", min_value=0, value=30)
    precio_competencia = st.number_input("Precio de la competencia", min_value=0.0, value=8.0)
    año = st.number_input("Año de predicción", min_value=2019, max_value=2026, value=2023)
    historial_compras = st.number_input("Historial de compras promedio", min_value=0.0, value=100.0)
    cantidad_disponible = st.number_input("Cantidad disponible", min_value=0, value=50)

    # Codificar la categoría seleccionada
    categoria_codificada = label_encoder_categoria.transform([categoria_seleccionada])[0]

    # Preparar los datos de entrada para la predicción
    input_data = pd.DataFrame({
        'precio': [precio],
        'vida_util': [vida_util],
        'precio_competencia': [precio_competencia],
        'año': [año],
        'historial_compras': [historial_compras],
        'cantidad_disponible': [cantidad_disponible],
        'nombre_categoria_encoded': [categoria_codificada]
    })

    # Predicción
    prediction = model.predict(input_data)

    # Mostrar la predicción
    st.write(f"### Predicción de cantidad comprada: {prediction[0]:.0f}")