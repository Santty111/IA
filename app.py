import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Configuración de la interfaz en el entorno
st.set_page_config(page_title="Merkaton UIO - IA Predictor", layout="wide")

st.title("📦 Sistema Inteligente de Predicción de Demanda - Merkaton UIO")
st.subheader("FICA - Inteligencia Artificial I (Progreso 3)")

# 1. Carga automática del dataset del proyecto
@st.cache_data
def cargar_datos():
    # El archivo debe estar en la misma carpeta del script
    df = pd.read_excel('dataset proyecto.xlsx')
    return df

try:
    df = cargar_datos()
    st.sidebar.success("✅ Dataset cargado correctamente (120 registros diarios).")
except Exception as e:
    st.sidebar.error(f"❌ Error al cargar 'dataset proyecto.xlsx': {e}")
    st.stop()

# 2. Pipeline y Entrenamiento de Modelos
X = df[['Precio_Unit', 'Lead_Time', 'Stock_Seg', 'Festividad', 'Estado_Vias', 'Criticidad']]
y = df['Ventas_Unidades']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Entrenar ambos grupos experimentales
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# 3. Panel de Control Lateral: Selección de Variables (Manipulación del Entorno)
st.sidebar.header("🎛️ Manipulación de Variables (Simulación)")
input_precio = st.sidebar.slider("Precio Unitario (USD)", float(df['Precio_Unit'].min()), float(df['Precio_Unit'].max()), float(df['Precio_Unit'].mean()))
input_lead = st.sidebar.slider("Lead Time Proveedor (Días)", int(df['Lead_Time'].min()), int(df['Lead_Time'].max()), int(df['Lead_Time'].mean()))
input_stock = st.sidebar.slider("Stock de Seguridad (Unidades)", int(df['Stock_Seg'].min()), int(df['Stock_Seg'].max()), int(df['Stock_Seg'].mean()))

input_festividad = st.sidebar.selectbox("¿Es día Festivo/Feriado en Quito?", options=[0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
input_vias = st.sidebar.selectbox("Estado de Vías de Acceso", options=[0, 1], format_func=lambda x: "Abiertas (1)" if x == 1 else "Bloqueadas/Cerradas (0)")
input_criticidad = st.sidebar.selectbox("Categoría de Criticidad del Producto", options=[1, 2, 3], format_func=lambda x: {1: "Vital (1)", 2: "Esencial (2)", 3: "Deseable (3)"}[x])

# 4. Despliegue de Resultados de la IA
col1, col2 = st.columns(2)

with col1:
    st.header("📊 Comparativa de Métricas de Éxito")
    metrics_df = pd.DataFrame({
        "Indicador (Métrica)": ["Coeficiente R² (Precisión)", "Error Absoluto Medio (MAE)"],
        "Grupo B: Gradient Boosting": [f"{r2_score(y_test, gb_model.predict(X_test)):.4f}", f"{mean_absolute_error(y_test, gb_model.predict(X_test)):.2f} unidades"],
        "Grupo A: Random Forest": [f"{r2_score(y_test, rf_model.predict(X_test)):.4f}", f"{mean_absolute_error(y_test, rf_model.predict(X_test)):.2f} unidades"]
    })
    st.table(metrics_df)
    
    st.info("""
    **Criterio del Indicador:** * El $R^2$ explica el porcentaje de varianza capturado.
    * El $MAE$ penaliza el error lineal en unidades físicas.
    """)

with col2:
    st.header("🔮 Predicción en Tiempo Real")
    
    # Vector de entrada para la inferencia
    features_input = np.array([[input_precio, input_lead, input_stock, input_festividad, input_vias, input_criticidad]])
    
    pred_gb = gb_model.predict(features_input)[0]
    pred_rf = rf_model.predict(features_input)[0]
    
    st.metric(label="Predicción Demanda - Gradient Boosting (Recomendado)", value=f"{int(pred_gb)} Unidades")
    st.metric(label="Predicción Demanda - Random Forest", value=f"{int(pred_rf)} Unidades")

# 5. Tabla Dinámica del Dataset
st.header("📋 Historial del Dataset Registrado")
st.dataframe(df)