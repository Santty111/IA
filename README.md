# Sistema Inteligente de Predicción de Demanda - Merkaton UIO

Este repositorio contiene un sistema predictivo inteligente desarrollado para optimizar la demanda diaria en **Merkaton UIO** (Quito), analizando variables operativas y del entorno urbano para predecir ventas y prevenir desabastecimiento.

## 🚀 Tecnologías Utilizadas

* **Python 3.13**
* **Streamlit** (Interfaz interactiva en tiempo real)
* **Scikit-Learn** (Algoritmos de aprendizaje automático supervisado)
* **Pandas & NumPy** (Carga y procesamiento de datos estructurados)

---

## 📊 Modelos Comparados (Grupo Experimental)

El sistema compara dos enfoques para determinar el modelo óptimo:

1. **Random Forest Regressor** ($R^2 \approx 0.835$)
2. **Gradient Boosting Regressor** ($R^2 \approx 0.930$) — *Modelo Recomendado por mayor precisión y menor error absoluto.*

---

## ⚙️ Cómo Ejecutar el Proyecto

1. Asegúrate de tener Python instalado.
2. Instala los paquetes requeridos:
   ```bash
   pip install streamlit pandas numpy scikit-learn openpyxl
   ```
3. Ejecuta la aplicación de Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Abre la dirección local [http://localhost:8501](http://localhost:8501) en tu navegador.
