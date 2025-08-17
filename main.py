import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Dataset
data = [
    ["Toyota", "Corolla", "Sedán", "Gasolina", 15, 20000, 470, "FWD", "Básica"],
    ["Ford", "Ranger", "Pickup", "Diesel", 11, 35000, 900, "4x4", "Intermedia"],
    ["Tesla", "Model 3", "Sedán", "Eléctrico", 0, 40000, 425, "RWD", "Alta"],
    ["Chevrolet", "Spark", "Hatchback", "Gasolina", 18, 12000, 185, "FWD", "Básica"],
    ["Hyundai", "Tucson", "SUV", "Gasolina", 12, 28000, 620, "AWD", "Intermedia"],
    ["Kia", "Rio", "Sedán", "Gasolina", 16, 15000, 390, "FWD", "Básica"],
    ["BMW", "X5", "SUV", "Diesel", 9, 60000, 750, "AWD", "Alta"],
    ["Audi", "A4", "Sedán", "Gasolina", 13, 45000, 480, "AWD", "Alta"],
    ["Nissan", "Leaf", "Hatchback", "Eléctrico", 0, 32000, 380, "FWD", "Intermedia"],
    ["Jeep", "Wrangler", "SUV", "Gasolina", 10, 42000, 1000, "4x4", "Alta"],
    ["Honda", "Civic", "Sedán", "Gasolina", 14, 25000, 450, "FWD", "Intermedia"],
    ["Mazda", "CX-5", "SUV", "Gasolina", 12, 30000, 550, "AWD", "Intermedia"],
    ["Renault", "Duster", "SUV", "Diesel", 14, 21000, 475, "FWD", "Básica"],
    ["Mercedes-Benz", "GLC", "SUV", "Gasolina", 11, 55000, 600, "AWD", "Alta"],
    ["Volkswagen", "Golf", "Hatchback", "Hibrido", 17, 23000, 380, "FWD", "Intermedia"]
]

df = pd.DataFrame(data, columns=["Marca","Modelo","Tipo","Combustible","Consumo","Precio","Maletero","Tracción","Tecnología"])

# Inputs de usuario
st.title("Te recomendamos tu auto ideal")
tipo = st.selectbox("Tipo de auto", df["Tipo"].unique())
combustible = st.selectbox("Combustible", df["Combustible"].unique())
consumo = st.slider("Consumo mínimo (km/L)", 0, 20, 14)
precio = st.number_input("Precio máximo", value=25000)
maletero = st.number_input("Capacidad mínima del maletero (L)", value=400)

# Filtrar dataset
df_filtrado = df[(df["Tipo"] == tipo) & (df["Combustible"] == combustible)]


#Sistema
if st.button("Recomendar"):
    if df_filtrado.empty:
        st.warning("No hay autos que cumplan esas características.")
    else:
        scaler = MinMaxScaler()
        df_scaled = df_filtrado.copy()
        df_scaled[["Consumo","Precio","Maletero"]] = scaler.fit_transform(df_filtrado[["Consumo","Precio","Maletero"]])

        usuario_vector = np.array([[consumo, precio, maletero]])
        usuario_vector = scaler.transform(usuario_vector)

        autos_vector = df_scaled[["Consumo","Precio","Maletero"]].values
        sim_scores = cosine_similarity(usuario_vector, autos_vector)[0]

        df_filtrado["Puntaje"] = sim_scores
        st.dataframe(df_filtrado.sort_values(by="Puntaje", ascending=False))