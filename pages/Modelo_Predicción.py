import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Configuración de página y estilo
st.set_page_config(layout="wide", page_title="Modelo de Predicción", page_icon="🔮")
st.title("Modelo de Predicción con Random Forest 🔜")
st.write("Este modelo se utiliza para predecir la cantidad comprada 🛒")
# Cambiar el fondo del dashboard a naranja y otros estilos
st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFF;  /* Fondo blanco */
    }
    .stSidebar, .stSidebar > div {
        background-color: #E0FFFF;  /* Color azul claro para la barra lateral */
    }
    .stApp {
        background-color: #FFFFFF;  /* Fondo blanco para la aplicación */
    }
    h1 {
        background-color: skyblue;
        color: #FFFFFF;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    h2, h3, h4, h5, h6 {
        background-color: lightblue;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    [data-testid="stFullScreenFrame"] {
        background-color: lightsalmon;
        text-align: center;
        border-top: 3px;
        border-bottom: 3px;
        border-left: 3px;
        border-radius: 15px 0px 0px 15px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Función para detectar valores atípicos usando el rango intercuartil (IQR)
def eliminar_valores_atipicos(df, column):
    # Calcular el cuartil 1 (Q1) y cuartil 3 (Q3)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Definir límites para detectar los outliers
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    # Filtrar los datos que están dentro de los límites
    df_sin_atipicos = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]
    
    return df_sin_atipicos

# Cargar los datos desde un archivo Excel
data = pd.read_excel("D:/Documentos/VALERIA/Documents/Univalle 4 Vale/Proyecto Integrador/PilAndina/PilAndina/df/copia.xlsx")

# Crear el DataFrame
df = pd.DataFrame(data)


# Convertir la columna 'año_mes' a formato de fecha para facilitar la manipulación


# Filtrar el DataFrame por los valores seleccionados por el usuario
st.sidebar.header("Filtrar datos")
nombre_tipo_input = st.sidebar.selectbox("Seleccione Nombre Tipo", df['nombre_tipo'].unique())
nombre_categoria_input = st.sidebar.selectbox("Seleccione Nombre Categoria", df['nombre_categoria'].unique())

# Filtrar los datos según la entrada del usuario
filtered_df = df[
                 (df['nombre_tipo'] == nombre_tipo_input) &
                 (df['nombre_categoria'] == nombre_categoria_input)]

col1,col2=st.columns((2))

with col1:
    # Mostrar el DataFrame en Streamlit
    st.subheader("Datos Originales")                                                                                               
    st.write(df)
with col2:
    # Mostrar el DataFrame filtrado
    st.subheader("Datos filtrados por Tipo")
    st.write(filtered_df)

# Verificar si el DataFrame filtrado tiene suficientes datos para el modelo
if len(filtered_df) >= 2:  # Para que se pueda realizar una regresión polinómica
    # Definir las variables independientes (X) y la dependiente (y)
    # Convertir 'año_mes' a una variable numérica que el modelo pueda usar (por ejemplo, convertirla a número de mes)
    filtered_df['año_mes_num'] = filtered_df['año_mes'].dt.month + 12 * (filtered_df['año_mes'].dt.year - filtered_df['año_mes'].dt.year.min())
    filtered_df = filtered_df.sort_values(by='año_mes_num', ascending=False)
    
    # Eliminar outliers usando el rango intercuartílico (IQR)
    Q1 = filtered_df['cantidad_comprada'].quantile(0.25)
    Q3 = filtered_df['cantidad_comprada'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrar los outliers en la variable dependiente
    filtered_df = filtered_df[(filtered_df['cantidad_comprada'] >= lower_bound) & (filtered_df['cantidad_comprada'] <= upper_bound)]
    

    X = filtered_df[['año_mes_num']]  # Año-Mes convertido a variable numérica
    y = filtered_df['cantidad_comprada']  # Cantidad comprada

    # Crear la transformación polinómica
    poly = PolynomialFeatures(degree=9)  # Puedes cambiar el grado según sea necesario
    X_poly = poly.fit_transform(X)

    # Crear el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_poly, y)

    # Realizar las predicciones
    y_pred = model.predict(X_poly)

    # Graficar los resultados
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', label='Datos reales')
    ax.plot(X, y_pred, color='red', label='Regresión Polinómica')

    # Etiquetas
    ax.set_title('Relación entre Año-Mes Numérico y Cantidad Comprada con Ajuste Polinómico')
    ax.set_xlabel('Año-Mes Numérico')
    ax.set_ylabel('Cantidad Comprada')
    ax.legend()

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Realizar las predicciones
    y_pred_test = model.predict(X_test)

    # Calcular las métricas de evaluación
    mae = mean_absolute_error(y_test, y_pred_test)  # Error absoluto medio
    mse = mean_squared_error(y_test, y_pred_test)  # Error cuadrático medio
    rmse = np.sqrt(mse)  # Raíz del error cuadrático medio
    r2 = r2_score(y_test, y_pred_test)  # Coeficiente de determinación (R²)

    # Mostrar las métricas en Streamlit
   # Crear un diccionario con las métricas
    metricas = {
        "Métrica": ["Error Absoluto Medio (MAE)", "Error Cuadrático Medio (MSE)", "Raíz del Error Cuadrático Medio (RMSE)", "Coeficiente de Determinación (R²)"],
        "Valor": [mae, mse, rmse, r2]
    }

    # Convertir el diccionario a un DataFrame
    df_metricas = pd.DataFrame(metricas)

    # Convertir el DataFrame a una lista de listas para usar con st.table
    tabla_sin_indices = df_metricas.values.tolist()

    # Agregar encabezados de columna
    st.write("### Evaluación del Modelo")
    st.table(pd.DataFrame(tabla_sin_indices, columns=["Métrica", "Valor"]))

    

    # Predicción para ene-23 
    prediction_input = np.array([[49]])
    prediction_input_poly = poly.transform(prediction_input)
    prediction_ene_23 = model.predict(prediction_input_poly)

    # Mostrar la predicción 
    st.write("### Predicción para enero de 2023")
    st.write(f"La predicción de cantidad comprada para Enero de 2023 es: {prediction_ene_23[0]:.2f}")

else:
    st.write("No hay suficientes datos para entrenar el modelo con los filtros seleccionados.")
