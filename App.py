# -*- coding: utf-8 -*-
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import datetime

# =============================
# 🎨 Estilo e interfaz católica (nueva paleta)
# =============================
st.set_page_config(page_title="✝️ Reconocimiento de Talentos", page_icon="🕊️", layout="centered")

st.markdown("""
<style>
  :root{
    --sky:#EAF3FB;        /* azul cielo claro */
    --light:#FFE797;      /* blanco marfil */
    --gold:#D9B95B;       /* oro suave */
    --navy:#1E3A5F;       /* azul profundo */
    --ink:#1A2A4F;        /* texto gris oscuro */
  }
  html, body, .stApp{
    background: radial-gradient(800px 500px at 10% 0%, var(--sky) 0%, var(--light) 80%);
    color: var(--ink) !important;
  }
  h1, h2, h3, h4, h5, h6{
    color: var(--navy);
    font-family: "Crimson Text","Georgia",serif;
    letter-spacing:.3px;
  }
  .stButton>button{
    background: linear-gradient(90deg, var(--gold), #F4E2A4) !important;
    color:#2C2C2C !important;
    border:none !important;
    border-radius:12px !important;
    font-weight:700 !important;
    box-shadow:0 2px 10px rgba(217,185,91,.35);
  }
  .stButton>button:hover{
    filter:brightness(1.05);
    box-shadow:0 3px 14px rgba(217,185,91,.45);
  }
  [data-testid="stSidebar"]{
    background: #F7FAFF !important;
    border-left: 2px solid #1A2A4F !important;
  }
  .blessing{
    padding:12px;
    border-left:4px solid var(--gold);
    background:#FFFDF5;
    border-radius:6px;
  }
</style>
""", unsafe_allow_html=True)

# =============================
# 🧠 Modelo de predicción
# =============================
@st.cache_resource
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype="float32") / 255.0
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# =============================
# 🕊️ Encabezado
# =============================
st.title("🕊️ Reconocimiento de Talentos — *Luz y Sabiduría Divina*")
st.markdown("""
Dios nos regaló la capacidad de pensar, crear y reconocer.  
Esta aplicación no solo lee números, sino que nos recuerda que **toda inteligencia es don del Creador.**

_"El temor del Señor es el principio de la sabiduría."_ (Proverbios 9,10)
""")

st.divider()
st.subheader("✍️ Dibuja un número y ofrece tu mente a Dios con gratitud")

# =============================
# ✏️ Lienzo de dibujo
# =============================
drawing_mode = "freedraw"
stroke_width = st.slider("✏️ Grosor del trazo", 1, 30, 12)
stroke_color = "#1E3A5F"  # azul profundo
bg_color = "#FFFFFf"      # fondo blanco luminoso

canvas_result = st_canvas(
    fill_color="rgba(255,255,255,0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=250,
    width=250,
    drawing_mode=drawing_mode,
    key="canvas",
)

# =============================
# 📖 Botón de predicción
# =============================
if st.button("✨ Reflexionar sobre mi trazo"):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
        input_image.save("prediction_digit.png")

        res = predictDigit(input_image)
        st.success(f"El número que trazaste es: **{res}** 🌟")

        # Reflexión católica por número
        reflexiones = {
            0: ("El origen", "Dios es el principio de todo. En Él inicia y termina nuestra historia."),
            1: ("Unidad", "Así como hay un solo Dios, busca la comunión y la paz."),
            2: ("Fraternidad", "El Señor envió a sus discípulos de dos en dos. Nunca estás solo."),
            3: ("La Trinidad", "Padre, Hijo y Espíritu Santo te envuelven en amor eterno."),
            4: ("Estabilidad", "Dios es la roca firme sobre la que construyes tu vida."),
            5: ("Gracia", "Tus manos y sentidos son dones que Dios te confía para hacer el bien."),
            6: ("Servicio", "El sexto día Dios creó al ser humano: trabaja con amor y dignidad."),
            7: ("Descanso", "El séptimo día Dios descansó: recuerda que el alma también necesita paz."),
            8: ("Nuevo comienzo", "El octavo día representa la Resurrección. Confía en lo nuevo que Dios hará."),
            9: ("Misión", "Jesús envió a sus discípulos al mundo. Anuncia su amor donde estés.")
        }

        titulo, mensaje = reflexiones.get(res, ("Misterio", "Dios obra de formas que trascienden nuestra comprensión."))
        st.markdown(f"### 🔢 {titulo}")
        st.markdown(f"_{mensaje}_")

        st.markdown("> “Haz todo por amor, nada por fuerza.” — San Francisco de Sales")
        hoy = datetime.date.today().strftime("%d %b %Y")
        st.markdown(f"<div class='blessing'>Que el Señor bendiga tu mente, tus dones y tu paz interior ☀️<br><em>{hoy}</em></div>", unsafe_allow_html=True)
    else:
        st.warning("✋ Por favor, dibuja un número antes de continuar.")

# =============================
# 📜 Sidebar (información)
# =============================
st.sidebar.title("📜 Acerca de esta aplicación")
st.sidebar.markdown("""
**“Reconocimiento de Talentos — Luz y Sabiduría Divina”**  
Esta app combina el aprendizaje automático con la reflexión cristiana.

💡 *Porque la inteligencia también puede ser oración si la usamos para servir.*

> “Dios ha puesto su sabiduría en el corazón del hombre.” (Job 38,36)
""")
st.sidebar.markdown("Desarrollado con fe y gratitud ✝️")





