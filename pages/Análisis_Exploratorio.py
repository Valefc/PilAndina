import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import seaborn as sns

st.set_page_config(layout="wide")

# Cambiar el fondo del dashboard a naranja
st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFF;  /* Color naranja */
    }
    .stSidebar, .stSidebar > div {
        background-color: #E0FFFF;  /* Color coral para la barra lateral */
    }
    .stApp {
        background-color:#FFFFFF;  /* Color naranja para la aplicación */
    }
    h1{
        background-color: skyblue;
        color: #FFFFFF;
        padding: 10px;
        border-radius: 10px;
        text-align:center;
        margin-bottom:20px;
        }
    h2,h3,h4,h5,h6{
        background-color: lightblue;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align:center;
        margin-bottom:20px;
        }
    [data-testid="stFullScreenFrame"]{
                background-color: lightsalmon;
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

df = pd.read_excel("df/BASE DE DATOS LECHE final oficial.xlsx", header = 0)

st.title("🔎Análisis Exploratorio de Datos Pil Andina S.A. 🥛")

col1,col2 =st.columns((2))
with col1:
    year=st.selectbox("Seleccione el año:", sorted(df["año"].unique()))


filtered_df=df[(df["año"]==year)].copy()
filtered_df_year=df[(df["año"]==year)].copy()

st.sidebar.header("Filtros")
select_categoria=st.sidebar.multiselect("Seleccione la/las Categorías", df['nombre_categoria'].unique(),default=df['nombre_categoria'].unique())
select_cliente=st.sidebar.selectbox("Seleccione el tipo de cliente",df['nombre_tipo'].unique())
select_variable=st.sidebar.selectbox("Seleccione una variable para las Gráficas",df.columns[2:])

df=df[(df["nombre_categoria"].isin(select_categoria)) & (df["nombre_tipo"])]
if st.checkbox("Mostrar información del dataset"):
    with st.expander("Estadísticas Descriptivas del Dataset"):
        st.subheader("Estadísticas Descriptivas del Dataset")
        st.write(df.describe())
    with st.expander("Datos Originales"):
        st.subheader("Datos Originales")
        st.write(df)

col1,col2=st.columns((2))

with col1:
    st.subheader(f"Distribución del {select_variable}")
    st.write(f"Este gráfico muestra la distribución del {select_variable} por Categoría")
    fig_gdp=px.histogram(df,x=select_variable,nbins=30,title=f"Distribución del {select_variable}",color_discrete_sequence=["cornflowerblue"])
    st.plotly_chart(fig_gdp)
    
with col2:
    st.subheader(f"Distribución del {select_variable}")
    st.write(f"Este gráfico representa la distribución del {select_variable} ")
    fig_gdp_capita=px.box(df,x="nombre_categoria",y=select_variable,title=f"Distibución del {select_variable}",color="nombre_categoria")
    st.plotly_chart(fig_gdp_capita)

with st.expander("💸 Indicadores económicos"):
    st.subheader("Gráficos de Barras Indicadores")
    fig_gdp2=px.bar(df,x="nombre_categoria",y=select_variable,color="nombre_categoria",barmode="overlay",title=f"Barras de Categoría vs {select_variable}")
    st.plotly_chart(fig_gdp2)


st.subheader(f"Precio por Producto")
fig_violin=px.violin(df, x="precio", y="tipo_promocion",title=f"Precio por Producto", color="tipo_promocion")
st.plotly_chart(fig_violin)

st.subheader(f"☁️ Nube de Palabras de Productos según {select_variable}")
fig_cloud=dict(zip(df['nombre_categoria'],df[select_variable]))
wordcloud=WordCloud(width=800, height=400, background_color='white')
wordcloud.generate_from_frequencies(fig_cloud)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis('off')
st.pyplot(plt)

st.subheader(f"📊 Gráfico de Barras {select_variable} por Producto")
df_barras = df.groupby('nombre_categoria')[select_variable].mean()

with st.expander(f"Datos Agrupados por {select_variable}"):
    st.write(df_barras)

fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(8, 4))
df_barras.plot(kind='bar', color='cornflowerblue') 
st.pyplot(fig)

# Crear un diccionario para convertir números de mes a nombres de mes
numero_a_mes = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

# Crear la columna month_year usando el mapeo
filtered_df_year["mes_nombre"] = filtered_df_year["mes"].map(numero_a_mes)
filtered_df_year["month_year"] = pd.to_datetime(filtered_df_year["año"].astype(str) + "-" + 
                                                 filtered_df_year["mes"].astype(str) + "-01")

# Sumar historial_compras por mes
line_chart = filtered_df_year.groupby("month_year")["historial_compras"].sum().reset_index()

line_chart = filtered_df_year.groupby("month_year")["historial_compras"].sum().reset_index()

fig_line = px.line(
    line_chart,
    x="month_year",
    y="historial_compras",
    labels={"historial_compras": "Cantidad"},
    line_shape='linear'
)
fig_line.update_traces(line_color="cornflowerblue") 

st.subheader(f"Serie para el historial de compras del año {year}")
st.plotly_chart(fig_line, use_container_width=True)

with st.expander("📊 Análisis de Correlación"):
    # Filtrar solo columnas numéricas
    df_numerico = df.select_dtypes(include=['int64', 'float64'])
    
    # Calcular la matriz de correlación
    correlation_matrix = df_numerico.corr()
    
    # Crear el gráfico de calor para la matriz de correlación
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    
    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)

