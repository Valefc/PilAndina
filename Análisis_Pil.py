import streamlit as st
import pandas as pd
import plotly_express as px
import os
import warnings

warnings.filterwarnings('ignore')

## Configuracion de la pagina
st.set_page_config(page_title='PIL ANDINA S.A. ', page_icon=":bar_chart", layout="wide")

st.title('ANÁLISIS PIL ANDINA S.A. ')

st.markdown(
    """
        <style>
            .stMetric{
                
                background-color:floralwhite;
                border: 1px solidad #E0E0E0;
                padding:10px;
                border-radius: 10px;
                box-shadow:2px 2px 5px rgba(0,0,0,0.1)
                
            }
        </style>
    """
    ,unsafe_allow_html= True     
    )
st.markdown(
    """
        <style>        
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
    """
    ,unsafe_allow_html= True     
    )

df = pd.read_excel("df/BASE DE DATOS LECHE final oficial.xlsx", header = 0)

col1, col2 = st.columns((2))
with col1:
    year = st.selectbox('Seleccione el año',df['año'].unique())
with col2:
    month=('Seleccione el mes')

#filtros

filtered_df = df[(df['año']==year)].copy()
filtered_df_year =  df[(df['año']==year)].copy()

st.sidebar.header('Escoge tu opcion')
categoria = st.sidebar.multiselect('Seleccione el producto',filtered_df['nombre_categoria'].unique())
filtered_df=filtered_df[filtered_df["nombre_categoria"].isin(categoria)]if categoria else filtered_df

cliente = st.sidebar.multiselect('Seleccione el cliente',filtered_df['nombre_tipo'].unique())
filtered_df=filtered_df[filtered_df["nombre_tipo"].isin(cliente)]if cliente else filtered_df

competencia = st.sidebar.multiselect('Seleccione la competencia',filtered_df['nombre_competencia'].unique())
filtered_df=filtered_df[filtered_df["nombre_competencia"].isin(competencia)] if competencia else filtered_df

col_a,col_b=st.columns((2))
with col_a:
    ingresos_totales=filtered_df["precio"].mean()
    st.metric("Media Compras (BS) de Productos Pil",f"Bs.{ingresos_totales:,.2f}")
with col_b:
    producto_df = filtered_df.groupby("nombre_categoria", as_index=False)["cantidad_comprada"].sum()
    producto_df = producto_df.sort_values(by="cantidad_comprada", ascending=False)
    producto_mas_vendido = producto_df.iloc[0]
    nombre_producto = producto_mas_vendido["nombre_categoria"]
    cantidad_producto = producto_mas_vendido["cantidad_comprada"]

    st.metric(
        f"Producto Más Vendido: {nombre_producto}",
        f"Unids: {cantidad_producto:,.2f}")

categoria_df=filtered_df.groupby(by=["nombre_categoria"], as_index=False)["precio"].sum()

col1,col2=st.columns ((2))
with col1:
    st.subheader("Precio por categoría")
    fig=px.bar(categoria_df,x="nombre_categoria",y="precio",text=['Bs{:,.2f}'.format(x) for x in categoria_df['precio']],
    template="seaborn",color="nombre_categoria")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("Historial de Compras por Tipo del Cliente")
    ciudad_df=filtered_df.groupby(by=["nombre_tipo"], as_index=False)["historial_compras"].sum()
    fig_pie=px.pie(ciudad_df, values="historial_compras",names="nombre_tipo",hole=0.5)
    st.plotly_chart(fig_pie, use_container_width=True)

cl1,cl2=st.columns((2))
with cl1:
    with st.expander("Ver datos de Producto"):
        st.write(categoria_df.style.background_gradient(cmap="PuBu"))
        csv=categoria_df.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar Datos",data=csv, file_name="Producto.csv",mime="text/csv", help="Haz clic aquí para ayuda")
with cl2:
    with st.expander("Ver datos de Cliente"):
        st.write(ciudad_df.style.background_gradient(cmap="PuBu"))
        csv=categoria_df.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar Datos",data=csv, file_name="Cliente.csv",mime="text/csv", help="Haz clic aquí para ayuda")



#Gráfico de Tree map
st.subheader("Vista Jerarquica de Ventas")
fig_tree=px.treemap(filtered_df,path=["nombre_tipo","tipo_promocion","nombre_categoria"],values="cantidad_comprada",hover_data=["cantidad_comprada"],color="nombre_categoria")
st.plotly_chart(fig_tree,use_container_width=True)


st.header("Gráfico de Dispersión Para los Precios VS Cantidad Comprada")
data1=px.scatter(filtered_df, x="cantidad_comprada", y="precio",size="cantidad_comprada",hover_name="nombre_categoria",color="nombre_categoria")
st.plotly_chart(data1,use_container_width=True)
