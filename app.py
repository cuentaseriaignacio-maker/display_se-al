import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
import streamlit as st

st.set_page_config(
    page_title="Colector FFT Virtual y Analizador de Señales DSP", layout="wide"
)

st.title("📟 Colector FFT Virtual & Analizador de Señales de Vibración (DSP)")
st.markdown("""
Esta herramienta simula el comportamiento de un **Colector de Datos / Analizador de Espectros Industrial**, 
permitiendo evaluar el efecto del muestreo, filtrado digital, conversión de variables físicas ($A \to V \to D$), 
promediado espectral con traslape (Overlap) y métricas globales ISO.
""")

# ==============================================================================
# 1. BARRA LATERAL: INGRESO DE SEÑAL Y CONFIGURACIÓN DEL SENSOR
# ==============================================================================
st.sidebar.header("1. 📥 Entrada de Señal y Sensor")

modo_entrada = st.sidebar.radio(
    "Origen de la Señal:",
    ["Generar Señal Sintética de Prueba", "Cargar Archivo (CSV / TXT)"],
)

if modo_entrada == "Cargar Archivo (CSV / TXT)":
  uploaded_file = st.sidebar.file_uploader(
      "Sube un archivo de tiempo (columna de tiempo y señal):",
      type=["csv", "txt"],
  )
  if uploaded_file is not None:
    try:
      df_in = pd.read_csv(uploaded_file)
      st.sidebar.success(f"Archivo cargado ({len(df_in)} muestras)")
      col_t = st.sidebar.selectbox("Columna de Tiempo [s]:", df_in.columns, index=0)
      col_y = st.sidebar.selectbox("Columna de Señal:", df_in.columns, index=1 if len(df_in.columns)>1 else 0)
      t_raw = df_in[col_t].values
      y_raw_in = df_in[col_y].values
      fs_estimada = int(1.0 / np.mean(np.diff(t_raw))) if len(t_raw) > 1 else 1000
    except Exception as e:
      st.sidebar.error(f"Error al leer archivo: {e}")
      t_raw = np.linspace(0, 2.0, 2000)
      y_raw_in = np.sin(2 * np.pi * 30 * t_raw)
      fs_estimada = 1000
  else:
    st.sidebar.info("Carga un archivo o usa la señal sintética.")
    t_raw = np.linspace(0, 2.0, 2000)
    y_raw_in = np.sin(2 * np.pi * 30 * t_raw)
    fs_estimada = 1000
else:
  # Generación Sintética de Prueba
  duracion = st.sidebar.slider("Duración de la Señal [s]:", 0.5, 10.0, 3.0, step=0.5)
  fs_estimada = st.sidebar.select_slider("Frecuencia de Muestreo fs [Hz]:", options=[512, 1024, 2048, 4096, 8192, 16384], value=2048)
  t_raw = np.linspace(0, duracion, int(duracion * fs_estimada), endpoint=False)
  
  # Componentes de prueba: 1X (25 Hz), 2X (50 Hz), Impacto de rodamiento (180 Hz) y ruido
  y_raw_in = (
      1.5 * np.sin(2 * np.pi * 25 * t_raw) + 
      0.8 * np.sin(2 * np.pi * 50 * t_raw) + 
      0.4 * np.sin(2 * np.pi * 180 * t_raw) * (np.sin(2 * np.pi * 5 * t_raw) > 0.5) +
      0.2 * np.random.normal(size=len(t_raw))
  )

st.sidebar.markdown("---")
unidad_inicial = st.sidebar.selectbox(
    " Magnitud Física Inicial de la Señal:",
    ["Aceleración [g]", "Aceleración [m/s²]", "Velocidad [mm/s]", "Desplazamiento [μm]", "Voltaje [mV]"]
)

if unidad_inicial == "Voltaje [mV]":
  sensibilidad = st.sidebar.number_input("Sensibilidad del Sensor [mV / g]:", value=100.0, step=10.0)
  y_raw_in = y_raw_in / sensibilidad  # Convertir a g
  unidad_base = "Aceleración [g]"
else:
  unidad_base = unidad_inicial

# ==============================================================================
# 2. CONFIGURACIÓN DEL COLECTOR FFT (MUESTREO Y LÍNEAS)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.header("2. ⚙️ Parámetros del Colector FFT")

num_lineas = st.sidebar.select_slider(
    "Número de Líneas Espectral (N_L):",
    options=[100, 200, 400, 800, 1600, 3200, 6400, 12800],
    value=800
)

# Relación Estándar N = 2.56 * N_L
N_bloque = int(2.56 * num_lineas)
fs = st.sidebar.number_input("Frecuencia de Muestreo Efectiva (fs) [Hz]:", value=int(fs_estimada), step=100)
f_max = fs / 2.56
delta_f = f_max / num_lineas

st.sidebar.info(f"""
* **Muestras por Bloque ($N$):** {N_bloque}
* **Frecuencia Máxima ($F_{{max}}$):** {f_max:.1f} Hz
* **Resolución ($\Delta f$):** {delta_f:.3f} Hz
* **Tiempo por Bloque ($T$):** {N_bloque/fs:.3f} s
""")

# ==============================================================================
# 3. TRUNCAMIENTO Y CORTE DE SEÑAL
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.header("3. ✂️ Recorte de Señal Temporal")

t_min_val, t_max_val = float(t_raw[0]), float(t_raw[-1])
rango_t = st.sidebar.slider(
    "Seleccionar Ventana Activa de Tiempo [s]:",
    min_value=t_min_val,
    max_value=t_max_val,
    value=(t_min_val, t_max_val),
    step=0.01
)

mask_activa = (t_raw >= rango_t[0]) & (t_raw <= rango_t[1])
t_act = t_raw[mask_activa]
y_act = y_raw_in[mask_activa]

# ==============================================================================
# 4. FILTRADO DIGITAL DSP (PASO ALTO, PASO BAJO, PASO BANDA)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.header("4. 🎛️ Filtros Digitales (DSP)")

tipo_filtro = st.sidebar.selectbox("Tipo de Filtro:", ["Sin Filtro", "Paso Alto (High-pass)", "Paso Bajo (Low-pass)", "Paso Banda (Band-pass)"])

f_c1, f_c2 = 0.0, 0.0
y_filtrada = y_act.copy()

if tipo_filtro != "Sin Filtro":
  orden = st.sidebar.slider("Orden del Filtro Butterworth:", 1, 8, 4)
  nyquist = 0.5 * fs
  
  if tipo_filtro == "Paso Alto (High-pass)":
    f_c1 = st.sidebar.slider("Frecuencia de Corte [Hz]:", 1.0, float(f_max), 10.0)
    b, a = butter(orden, f_c1 / nyquist, btype='high')
    y_filtrada = filtfilt(b, a, y_act)
    
  elif tipo_filtro == "Paso Bajo (Low-pass)":
    f_c1 = st.sidebar.slider("Frecuencia de Corte [Hz]:", 5.0, float(f_max), 200.0)
    b, a = butter(orden, f_c1 / nyquist, btype='low')
    y_filtrada = filtfilt(b, a, y_act)
    
  elif tipo_filtro == "Paso Banda (Band-pass)":
    col_f1, col_f2 = st.sidebar.columns(2)
    f_c1 = col_f1.number_input("F_Corte Inf [Hz]:", value=10.0, min_value=1.0)
    f_c2 = col_f2.number_input("F_Corte Sup [Hz]:", value=200.0, max_value=float(f_max))
    b, a = butter(orden, [f_c1 / nyquist, f_c2 / nyquist], btype='band')
    y_filtrada = filtfilt(b, a, y_act)

# ==============================================================================
# 5. CONVERSIÓN DE VARIABLE FÍSICA (ACELERACIÓN / VELOCIDAD / DESPLAZAMIENTO)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.header("5. 🔄 Conversión de Variable Física")

var_salida = st.sidebar.selectbox(
    "Visualizar Señal Convertida en:",
    ["Aceleración [g]", "Aceleración [m/s²]", "Velocidad [mm/s]", "Desplazamiento [μm]"]
)

def convertir_senal(y_in, fs_val, unidad_origen, unidad_destino):
  # Transforma la señal en el dominio espectral
  N_len = len(y_in)
  Y_fft = rfft(y_in)
  freqs = rfftfreq(N_len, 1/fs_val)
  
  # Evitar división por cero en DC
  freqs_safe = freqs.copy()
  freqs_safe[0] = 1e-6
  omega = 2 * np.pi * freqs_safe
  
  # 1. Normalizar origen a Aceleración m/s²
  if "g" in unidad_origen:
    a_m_s2 = Y_fft * 9.81
  elif "m/s²" in unidad_origen:
    a_m_s2 = Y_fft.copy()
  elif "mm/s" in unidad_origen:
    a_m_s2 = Y_fft * (1j * omega) / 1000.0
  elif "μm" in unidad_origen:
    a_m_s2 = Y_fft * ((1j * omega)**2) / 1e6
  else:
    a_m_s2 = Y_fft.copy()
    
  # Filtro anti-rumble para bajas frecuencias (< 2 Hz)
  mask_low = freqs < 2.0
  
  # 2. Convertir a destino
  if unidad_destino == "Aceleración [g]":
    Y_out = a_m_s2 / 9.81
  elif unidad_destino == "Aceleración [m/s²]":
    Y_out = a_m_s2
  elif unidad_destino == "Velocidad [mm/s]":
    Y_out = (a_m_s2 / (1j * omega)) * 1000.0
    Y_out[mask_low] = 0.0
  elif unidad_destino == "Desplazamiento [μm]":
    Y_out = (a_m_s2 / ((1j * omega)**2)) * 1e6
    Y_out[mask_low] = 0.0
    
  y_time_converted = np.fft.irfft(Y_out, n=N_len)
  return y_time_converted, freqs, np.abs(Y_out) * (2.0 / N_len)

y_procesada, freqs_espectro, amp_espectro = convertir_senal(y_filtrada, fs, unidad_base, var_salida)

# ==============================================================================
# 6. MÓDULO DE PROMEDIADO ESPESTRAL Y TRASLAPE (OVERLAP)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.header("6. 📊 Promediado Espectral y Traslape")

hab_promediado = st.sidebar.checkbox("Habilitar Promediado Espectral", value=True)

if hab_promediado:
  tipo_promedio = st.sidebar.selectbox("Tipo de Promedio:", ["Promedio Lineal / Normal", "Peak Hold (Mantener Máximos)", "Promedio Exponencial / Ponderado"])
  overlap_pct = st.sidebar.select_slider("Porcentaje de Traslape (Overlap):", options=[0, 25, 50, 75], value=50)
  marcar_bloques = st.sidebar.checkbox("Marcar Bloques de Traslape en Tiempo", value=True)
  
  # Cálculo de saltos de bloque
  paso = int(N_bloque * (1.0 - overlap_pct / 100.0))
  num_muestras_totales = len(y_procesada)
  
  if num_muestras_totales >= N_bloque:
    K_max = int((num_muestras_totales - N_bloque) / paso) + 1
  else:
    K_max = 0
    
  K_promedios = st.sidebar.number_input(f"Número de Promedios (Máx disponible: {K_max}):", min_value=1, max_value=max(1, K_max), value=max(1, min(8, K_max)))
  
  # Algoritmo de promediado por bloques
  espectros_bloques = []
  limites_bloques = []
  
  w_hanning = np.hanning(N_bloque)
  
  for k in range(K_promedios):
    i_start = k * paso
    i_end = i_start + N_bloque
    if i_end <= num_muestras_totales:
      bloque_t = y_procesada[i_start:i_end] * w_hanning
      fft_b = np.abs(rfft(bloque_t)) * (2.0 / N_bloque) * 1.63 # Corrección de amplitud Hanning
      espectros_bloques.append(fft_b)
      limites_bloques.append((t_act[i_start], t_act[i_end-1]))
      
  freqs_prom = rfftfreq(N_bloque, 1/fs)
  
  if len(espectros_bloques) > 0:
    matriz_esp = np.array(espectros_bloques)
    if tipo_promedio == "Promedio Lineal / Normal":
      amp_promedidada = np.mean(matriz_esp, axis=0)
    elif tipo_promedio == "Peak Hold (Mantener Máximos)":
      amp_promedidada = np.max(matriz_esp, axis=0)
    else: # Exponencial
      weights = np.exp(np.linspace(-1, 0, K_promedios))
      weights /= np.sum(weights)
      amp_promedidada = np.average(matriz_esp, axis=0, weights=weights)
  else:
    freqs_prom = freqs_espectro
    amp_promedidada = amp_espectro
    limites_bloques = []
else:
  marcar_bloques = False
  freqs_prom = freqs_espectro
  amp_promedidada = amp_espectro
  limites_bloques = []

# ==============================================================================
# 7. VALORES GLOBALES (OVERALL) Y SUB-RANGO PERSONALIZADO
# ==============================================================================
st.markdown("### 1. 📏 Valores Globales de Vibración (Overall)")

def calcular_metricas(sig):
  rms = np.sqrt(np.mean(sig**2))
  peak = np.max(np.abs(sig))
  p2p = np.max(sig) - np.min(sig)
  return rms, peak, p2p

rms_act, peak_act, p2p_act = calcular_metricas(y_procesada)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric(f"RMS Global ({var_salida})", f"{rms_act:.3f}")
col_m2.metric(f"Pico (Peak)", f"{peak_act:.3f}")
col_m3.metric(f"Pico a Pico (P-P)", f"{p2p_act:.3f}")
col_m4.metric("Factor de Cresta", f"{(peak_act/rms_act if rms_act>0 else 0):.2f}")

# Sub-rango personalizado
st.markdown("---")
hab_subrango = st.checkbox("🔍 Habilitar Cálculo de Valores Globales en Sub-Rango Personalizado")

if hab_subrango:
  col_sr1, col_sr2 = st.columns(2)
  tipo_subrango = col_sr1.radio("Filtrar Sub-Rango por:", ["Dominio del Tiempo [s]", "Dominio de la Frecuencia [Hz]"])
  
  if tipo_subrango == "Dominio del Tiempo [s]":
    sr_t1, sr_t2 = col_sr2.slider("Seleccionar Rango de Tiempo [s]:", float(t_act[0]), float(t_act[-1]), (float(t_act[0]), float(t_act[-1])))
    mask_sr = (t_act >= sr_t1) & (t_act <= sr_t2)
    sig_sr = y_procesada[mask_sr]
  else:
    sr_f1, sr_f2 = col_sr2.slider("Seleccionar Banda de Frecuencia [Hz]:", 0.0, float(f_max), (0.0, float(f_max)))
    mask_sr_f = (freqs_prom >= sr_f1) & (freqs_prom <= sr_f2)
    # Parseval: RMS en frecuencia
    sig_sr_rms = np.sqrt(np.sum(amp_promedidada[mask_sr_f]**2) / 2.0)
    sig_sr = np.array([sig_sr_rms, sig_sr_rms*np.sqrt(2)]) # Estimación aproximada
    
  if len(sig_sr) > 0:
    rms_sr, peak_sr, p2p_sr = calcular_metricas(sig_sr)
    st.info(f"**Métricas del Sub-Rango Seleccionado:** RMS = **{rms_sr:.3f}** | Peak = **{peak_sr:.3f}** | P-P = **{p2p_sr:.3f}**")

# ==============================================================================
# 8. DESPLIEGUE VISUAL DE GRÁFICAS (TIEMPO Y ESPECTRO)
# ==============================================================================
st.markdown("---")
st.markdown("### 2. 📉 Visualización en Tiempo y Frecuencia")

fig_master = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        f"a) Señal Temporal Processada ({var_salida}) — [Muestra original descartada en gris punteado]",
        f"b) Espectro de Frecuencia FFT ({var_salida}) — [Líneas de Control $f_s, F_{{max}}, f_c$]"
    ),
    vertical_spacing=0.15
)

# 1. Gráfico Temporal: Señal Descartada (Gris) + Señal Activa
fig_master.add_trace(
    go.Scatter(x=t_raw, y=y_raw_in, mode='lines', name='Señal Descartada/Original', line=dict(color='lightgray', dash='dot', width=1.5)),
    row=1, col=1
)
fig_master.add_trace(
    go.Scatter(x=t_act, y=y_procesada, mode='lines', name='Señal Activa Procesada', line=dict(color='#1f77b4', width=2)),
    row=1, col=1
)

# Marcar bloques de traslape en el gráfico de tiempo
if marcar_bloques and len(limites_bloques) > 0:
  colors_bloques = ['rgba(255, 127, 14, 0.2)', 'rgba(44, 160, 44, 0.2)', 'rgba(214, 39, 40, 0.2)', 'rgba(148, 103, 189, 0.2)']
  for idx_b, (tb_start, tb_end) in enumerate(limites_bloques):
    col_b = colors_bloques[idx_b % len(colors_bloques)]
    fig_master.add_vrect(
        x0=tb_start, x1=tb_end, fillcolor=col_b, opacity=0.5,
        layer="below", line_width=1.5, line_dash="dash",
        annotation_text=f"B{idx_b+1}", annotation_position="top left",
        row=1, col=1
    )

# 2. Gráfico Espectral FFT
fig_master.add_trace(
    go.Scatter(x=freqs_prom[freqs_prom <= f_max], y=amp_promedidada[freqs_prom <= f_max], mode='lines', name='Espectro FFT', line=dict(color='crimson', width=2)),
    row=2, col=1
)

# Marcadores de Control de Frecuencia
fig_master.add_vline(x=f_max, line_dash="dash", line_color="orange", annotation_text=f"Fmax = {f_max:.0f} Hz", row=2, col=1)
fig_master.add_vline(x=fs, line_dash="dot", line_color="red", annotation_text=f"fs = {fs} Hz", row=2, col=1)

if tipo_filtro != "Sin Filtro":
  if f_c1 > 0:
    fig_master.add_vline(x=f_c1, line_dash="solid", line_color="purple", annotation_text=f"Fc1 = {f_c1:.1f} Hz", row=2, col=1)
  if f_c2 > 0:
    fig_master.add_vline(x=f_c2, line_dash="solid", line_color="purple", annotation_text=f"Fc2 = {f_c2:.1f} Hz", row=2, col=1)

fig_master.update_layout(template="plotly_white", height=750)
fig_master.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
fig_master.update_xaxes(title_text="Frecuencia (Hz)", range=[0, f_max * 1.05], row=2, col=1)
fig_master.update_yaxes(title_text=var_salida, row=1, col=1)
fig_master.update_yaxes(title_text=f"Amplitud ({var_salida})", row=2, col=1)

st.plotly_chart(fig_master, use_container_width=True)
