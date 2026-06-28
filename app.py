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
    
    # Vector de entrada para la inferencia (usando DataFrame para evitar advertencias de nombres de columnas)
    features_input = pd.DataFrame(
        [[input_precio, input_lead, input_stock, input_festividad, input_vias, input_criticidad]], 
        columns=X.columns
    )
    
    pred_gb = gb_model.predict(features_input)[0]
    pred_rf = rf_model.predict(features_input)[0]
    
    # Contenedor de métricas lado a lado
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="Gradient Boosting (Recomendado)", value=f"{int(pred_gb)} Unidades")
    with col_m2:
        st.metric(label="Random Forest", value=f"{int(pred_rf)} Unidades")


# 4.5. Gráfico de Sensibilidad al Precio (Curva de Demanda Dinámica)
st.markdown("---")
st.header("📈 Curva de Demanda Dinámica (Sensibilidad al Precio)")
st.write("El siguiente gráfico simula la demanda esperada si variáramos únicamente el **Precio Unitario**, manteniendo las demás variables fijas según tu configuración en el panel lateral.")

# Generar un rango de precios desde el mínimo al máximo registrado
precios_simulados = np.linspace(float(df['Precio_Unit'].min()), float(df['Precio_Unit'].max()), 50)

# Construir DataFrame de prueba para evitar advertencias de nombres de columnas
sensitivity_features = pd.DataFrame(np.zeros((len(precios_simulados), 6)), columns=X.columns)
sensitivity_features['Precio_Unit'] = precios_simulados
sensitivity_features['Lead_Time'] = input_lead
sensitivity_features['Stock_Seg'] = input_stock
sensitivity_features['Festividad'] = input_festividad
sensitivity_features['Estado_Vias'] = input_vias
sensitivity_features['Criticidad'] = input_criticidad

# Predicciones simultáneas
y_pred_gb_sim = gb_model.predict(sensitivity_features)
y_pred_rf_sim = rf_model.predict(sensitivity_features)

# DataFrame para graficar
sensitivity_df = pd.DataFrame({
    "Precio Unitario (USD)": precios_simulados,
    "Gradient Boosting (Rec.)": y_pred_gb_sim,
    "Random Forest": y_pred_rf_sim
})

# Graficar curva de sensibilidad
st.line_chart(
    sensitivity_df,
    x="Precio Unitario (USD)",
    y=["Gradient Boosting (Rec.)", "Random Forest"],
    use_container_width=True
)


# 5. Tabla Dinámica del Dataset
st.header("📋 Historial del Dataset Registrado")
st.dataframe(df)