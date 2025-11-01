# -*- coding: utf-8 -*-
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import datetime

# =============================
# Estilo e interfaz católica
# =============================
st.set_page_config(page_title="✝️ Reconocimiento de Talentos", page_icon="🕊️", layout="centered")

st.markdown("""
<style>
  :root{
    --parchment:#F8F3E7;
    --ink:#3E2F1C;
    --gold:#C5A253;
    --maryblue:#274B8A;
  }
  html, body, .stApp{
    background: radial-gradient(900px 500px at 10% 0%, #fff9ee 0%, var(--parchment) 60%);
    color: var(--ink) !important;
  }
  h1, h2, h3, h4, h5, h6{
    color: var(--maryblue);
    font-family:"Crimson Text","Georgia",serif;
  }
  .stButton>button{
    background: linear-gradient(90deg,var(--gold),#e3c77a) !important;
    color:#3b2d12 !important;
    border:none !important;
    border-radius:10px !important;
    font-weight:700 !important;
    box-shadow:0 2px 10px rgba(197,162,83,.35);
  }
  .stButton>button:hover{filter:brightness(1.07);}
  [data-testid="stSidebar"]{
    background:#FAF6EC !important;
    border-left:1px solid #eadfc6 !important;
  }
  .blessing{
    padding:12px;
    border-left:4px solid var(--gold);
    background:#fffdf7;
    border-radius:6px;
  }
</style>
""", unsafe_allow_html=True)

# =============================
# Modelo de predicción
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
# Encabezado
# =============================
st.title("✝️ Reconocimiento de Talentos — *En tus manos, Señor*")
st.markdown("""
Dios nos ha dado una mente capaz de aprender, crear y reconocer.  
Hoy, esta inteligencia artificial te invita a **reflexionar sobre el don del conocimiento**,  
mientras reconoces un número trazado por tus propias manos.  

_"Porque el Señor da la sabiduría, de su boca brotan la ciencia y la prudencia."_ (Proverbios 2,6)
""")

st.divider()
st.subheader("✍️ Dibuja un número y deja que la sabiduría divina te inspire")

# =============================
# Lienzo
# =============================
drawing_mode = "freedraw"
stroke_width = st.slider("✏️ Grosor del trazo", 1, 30, 12)
stroke_color = "#FFFFFF"
bg_color = "#000000"

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
# Botón para predecir
# =============================
if st.button("📖 Reflexionar sobre mi trazo"):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
        input_image.save("prediction_digit.png")

        res = predictDigit(input_image)
        st.success(f"El número que tu mente ha trazado es: **{res}** ✨")

        # Reflexión católica según el número
        reflexiones = {
            0: ("El comienzo", "Todo inicia en Dios, que es el Alfa y el Omega."),
            1: ("La unidad", "El Señor es uno. Busca la comunión y no la división."),
            2: ("La cooperación", "Como los discípulos enviados de dos en dos, trabaja en comunidad."),
            3: ("La Trinidad", "Padre, Hijo y Espíritu Santo te acompañan en toda obra."),
            4: ("La firmeza", "Edifica tu vida sobre roca firme: la fe en Cristo."),
            5: ("La gracia", "El número cinco recuerda los dones de Dios que llenan tus manos."),
            6: ("El trabajo", "Dios bendijo el sexto día con la creación del ser humano. Trabaja con amor."),
            7: ("La plenitud", "El séptimo día Dios descansó. Aprende a confiar y reposar en Él."),
            8: ("La eternidad", "El ocho simboliza un nuevo comienzo: Cristo resucitado al amanecer."),
            9: ("La misión", "Jesús envió a los setenta y dos: lleva su palabra a donde vayas.")
        }

        titulo, mensaje = reflexiones.get(res, ("Misterio", "Dios obra en formas que no siempre comprendemos."))
        st.markdown(f"### 🔢 {titulo}")
        st.markdown(f"_{mensaje}_")

        st.markdown("""
        > “Que todo lo que hagas, lo hagas para gloria de Dios.” (1 Cor 10,31)
        """)
        hoy = datetime.date.today().strftime("%d %b %Y")
        st.markdown(f"<div class='blessing'>Que el Señor bendiga tu mente y tus manos 🕊️ <br><em>{hoy}</em></div>", unsafe_allow_html=True)
    else:
        st.warning("✋ Por favor, dibuja un número antes de continuar.")

# =============================
# Sidebar
# =============================
st.sidebar.title("🕊️ Acerca de esta aplicación")
st.sidebar.markdown("""
Esta aplicación fue inspirada en la idea de que **toda inteligencia proviene de Dios**  
y puede usarse para **servir, aprender y crecer en humildad**.

> “No tengan miedo a la ciencia, porque toda verdad viene de Dios.” — San Juan Pablo II  
""")
st.sidebar.markdown("Desarrollado con gratitud y fe ✝️")
