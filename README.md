# display_señal

# 📟 Colector Virtual FFT y Analizador de Señales DSP

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)
![SciPy](https://img.shields.io/badge/SciPy-DSP-blue.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Simulador y analizador interactivo de **Procesamiento Digital de Señales (DSP)** y **Análisis de Vibraciones en Maquinaria Rotatoria**. 

La aplicación emula las funciones avanzadas de un **Colector de Datos FFT Industrial** (como Emerson CSI, SKF Microlog o PDM), permitiendo evaluar configuraciones de muestreo, filtros digitales, conversión de unidades físicas y promediado espectral con traslape.

---

## 🚀 Acceso a la Aplicación Web

Puedes utilizar la herramienta en línea directamente en tu navegador:
👉 **[Abrir Colector Virtual en Streamlit Cloud](https://tu-usuario.streamlit.app)** *(reemplaza esta URL con el enlace de tu app)*

---

## 🕹️ Funcionalidades del Analizador

1. **Ingreso Flexible de Datos:**
   - Carga de archivos reales en formato `.csv` o `.txt`.
   - Generación de señal sintética de prueba con armónicos de rotación (\\(1\text{X}, 2\text{X}\\)), impactos de rodamiento y ruido blanco aleatorio.
   - Conversión de voltajes crudos (\\(mV\\)) a unidades de ingeniería según la sensibilidad del acelerómetro (\\(mV/g\\)).

2. **Seteo Digital de Colector FFT:**
   - Selección de líneas espectrales (\\(N_L\\) de 100 a 12,800 líneas).
   - Aplicación estricta de la relación \\(N = 2.56 \times N_L\\) y cálculo automático de la frecuencia de corte anti-aliasing \\(F_{max} = f_s / 2.56\\).

3. **Corte y Referencia Visual:**
   - Permite truncar la ventana de tiempo activa manteniendo la fracción descartada visible en **gris punteado** como referencia.

4. **Filtrado Digital DSP (Butterworth):**
   - Filtros **Paso Alto**, **Paso Bajo** y **Paso Banda** con selección de orden y frecuencia de corte.

5. **Conversión Dinámica de Variable Física (\\(A \to V \to D\\)):**
   - Transición instantánea entre Aceleración (\\(g\\) o \\(m/s^2\\)), Velocidad (\\(mm/s\\)) y Desplazamiento (\\(\mu m\\)) mediante integración espectral con filtro anti-rumble.

6. **Promediado Espectral con Traslape (Overlap):**
   - Modos: **Promedio Lineal**, **Peak Hold (Máximos)** y **Promedio Exponencial**.
   - Traslape configurable (0%, 25%, 50%, 75%).
   - Delimitación gráfica sobre la onda temporal mostrando las cajas/marcadors de cada bloque \\(B_1, B_2, \dots, B_K\\).

7. **Métricas Globales (Overall) y Sub-Rango:**
   - Cálculo automático de valores RMS, Peak, Peak-to-Peak y Factor de Cresta.
   - Herramienta de **Sub-Rango Personalizado** para evaluar métricas en bandas específicas de tiempo o frecuencia.

---

## 🛠️ Ejecución Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/colector-vibraciones-dsp.git
cd colector-vibraciones-dsp

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar la app
streamlit run app.py
