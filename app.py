import streamlit as st
import pandas as pd
import joblib
import json
import os

# Configuración de la página
st.set_page_config(page_title="Evaluador de Riesgo Actuarial", layout="wide")

st.title("📊 Sistema de Evaluación de Riesgo Actuarial - Seguros Médicos")
st.markdown("""
Esta aplicación utiliza modelos de Aprendizaje No Supervisado (K-Means) y Supervisado (SVM) 
para segmentar e identificar el nivel de riesgo de un cliente potencial.
""")

# Intentar cargar metadatos y modelos
@st.cache_resource
def load_models():
    try:
        # Ajusta las rutas según tu estructura de carpetas
        modelo_kmeans = joblib.load("models/kmeans_riesgo_actuarial.pkl")
        
        # Opcional: Cargar metadatos si existen
        metadata = None
        if os.path.exists("models/model_metadata.json"):
            with open("models/model_metadata.json", "r") as f:
                metadata = json.load(f)
                
        return modelo_kmeans, metadata
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None, None

modelo, meta = load_models()

# Crear el formulario en la barra lateral o panel principal
st.header("📋 Datos del Cliente")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("Sexo", options=["male", "female"])
    bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Número de Hijos", min_value=0, max_value=10, value=0, step=1)

with col2:
    smoker = st.selectbox("¿Es Fumador?", options=["no", "yes"])
    region = st.selectbox("Región Residencial", options=["southwest", "southeast", "northwest", "northeast"])
    charges = st.number_input("Cargos Médicos Históricos ($)", min_value=0.0, value=5000.0, step=100.0)

# Botón para realizar la predicción
if st.button("Evaluar Riesgo del Cliente"):
    if modelo is not None:
        # 1. Crear el DataFrame con la estructura idéntica que espera el Pipeline del notebook
        nuevo_cliente = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": charges
        }])
        
        # 2. Realizar predicción de Cluster
        try:
            # Si el pkl guardado es un Pipeline de Scikit-Learn que ya procesa/escala las columnas:
            cluster_predicho = modelo.predict(nuevo_cliente)[0]
            
            # Mapas de interpretación (Asegúrate de comprobar a qué riesgo corresponde cada cluster en tu notebook)
            # Ejemplo ilustrativo: Cluster 0=Bajo, 1=Medio, 2=Alto
            mapeo_riesgo = {0: "Bajo", 1: "Medio", 2: "Alto"}
            nivel_riesgo = mapeo_riesgo.get(cluster_predicho, "Desconocido")
            
            # 3. Mostrar resultados estéticos en la interfaz
            st.subheader("🎯 Resultado del Análisis")
            
            # Color dinámico según el riesgo asignado
            if nivel_riesgo == "Bajo":
                st.success(f"**Cluster Asignado:** {cluster_predicho} | **Nivel de Riesgo Actuarial:** BAJO")
                st.info("💡 **Explicación:** El perfil presenta bajos cargos médicos, no fumador o parámetros de salud estables.")
            elif nivel_riesgo == "Medio":
                st.warning(f"**Cluster Asignado:** {cluster_predicho} | **Nivel de Riesgo Actuarial:** MEDIO")
                st.info("💡 **Explicación:** El perfil presenta un nivel de riesgo moderado basado en edad o cargos médicos intermedios.")
            else:
                st.error(f"**Cluster Asignado:** {cluster_predicho} | **Nivel de Riesgo Actuarial:** ALTO")
                st.info("💡 **Explicación:** Factores como ser fumador o presentar cargos médicos muy elevados sitúan al perfil en riesgo alto.")
                
        except Exception as err:
            st.error(f"Ocurrió un error al procesar los datos en el modelo: {err}")
            st.info("Asegúrate de que las columnas pasadas coincidan exactamente con la transformación del pipeline de entrenamiento.")
    else:
        st.warning("El modelo no está disponible.")
