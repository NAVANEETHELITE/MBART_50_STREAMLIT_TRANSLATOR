import pytesseract as pt
from pathlib import Path
from PIL import Image
import io
import re
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from textblob import TextBlob
import streamlit as st

pt.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

st.markdown("<h1 style='color:white;'>MULTILINGUAL TRANSLATOR USING TRANSFORMERS</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='color:orange;'>HOW TO USE?</h3>", unsafe_allow_html=True)

st.text("1.UPLOAD IMAGE")
st.text("2.CLICK EXTRACT AND TRANSLATE TEXT")
st.text("3.CHOOSE DIFFERENT LANGUAGE AND AGAIN CLICK EXTRACT AND TRANSLATE")

st.sidebar.markdown("<h3 style='color:orange;'>UPLOAD IMAGE</h3>", unsafe_allow_html=True)
image_file = st.sidebar.file_uploader("UPLOAD A HIGH QUALITY IMAGE WITH PROPER FONT", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("<h3 style='color:orange;'>CONTACT DEVELOPER:</h3>", unsafe_allow_html=True)
st.sidebar.write("[LINKEDIN](https://www.linkedin.com/in/navaneethan-s-a527571b7/)")
st.sidebar.write("[EMAIL](mailto:navaneethanselvakumar@gmail.com)")
st.sidebar.write("[INSTAGRAM](https://www.instagram.com/_navneeth_/)")

inp_tran = ''
def extract(img):
    global inp_tran
    text = pt.image_to_string(img)
    text = re.sub('[^a-zA-Z0-9]', ' ',text)
    text = text.replace("  ", " ")
    if(text!=""):
        corrected = TextBlob(text).correct()
        inp_tran += str(corrected)
        st.markdown("<h2 style='color:orange;'>EXTRACTED TEXT:</h2>", unsafe_allow_html=True)
        st.markdown(f"{corrected}", unsafe_allow_html=True)
    else:
        st.markdown("""<h1 style='text-align: center;'>TEXTLESS IMAGE</h1>""",unsafe_allow_html=True)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def download_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = download_model()

if image_file is not None:
    st.markdown("<h2 style='color:orange;'>UPLOADED IMAGE:</h2>", unsafe_allow_html=True)
    st.image(image_file,width=400)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    if st.button('EXTRACT AND TRANSLATE TEXT'):
        extract(img)
    
    
def Translate(text, lang):
    lang_dict = {'Arabic' :'ar_AR', 
    'Czech' :'cs_CZ', 
    'German' :'de_DE', 
    'English': 'en_XX', 
    'Spanish' :'es_XX', 
    'Estonian' :'et_EE', 
    'Finnish' :'fi_FI', 
    'French' :'fr_XX', 
    'Gujarati' :'gu_IN', 
    'Hindi' :'hi_IN', 
    'Italian' :'it_IT', 
    'Japanese' :'ja_XX', 
    'Kazakh' :'kk_KZ', 
    'Korean' :'ko_KR', 
    'Lithuanian' :'lt_LT', 
    'Latvian' :'lv_LV', 
    'Burmese' :'my_MM', 
    'Nepali' :'ne_NP', 
    'Dutch' :'nl_XX', 
    'Romanian' :'ro_RO', 
    'Russian' :'ru_RU', 
    'Sinhala' :'si_LK', 
    'Turkish' :'tr_TR', 
    'Vietnamese' :'vi_VN', 
    'Chinese' :'zh_CN', 
    'Afrikaans' :'af_ZA', 
    'Azerbaijani' :'az_AZ', 
    'Bengali' :'bn_IN', 
    'Persian' :'fa_IR', 
    'Hebrew' :'he_IL', 
    'Croatian' :'hr_HR', 
    'Indonesian' :'id_ID', 
    'Georgian' :'ka_GE', 
    'Khmer' :'km_KH', 
    'Macedonian' :'mk_MK', 
    'Malayalam' :'ml_IN', 
    'Mongolian' :'mn_MN', 
    'Marathi' :'mr_IN', 
    'Polish' :'pl_PL', 
    'Pashto' :'ps_AF', 
    'Portuguese' :'pt_XX', 
    'Swedish' :'sv_SE', 
    'Swahili' :'sw_KE', 
    'Tamil' :'ta_IN', 
    'Telugu' :'te_IN', 
    'Thai' : 'th_TH', 
    'Tagalog' : 'tl_XX', 
    'Ukrainian' :'uk_UA', 
    'Urdu' : 'ur_PK', 
    'Xhosa' : 'xh_ZA', 
    'Galician' : 'gl_ES', 
    'Slovene' :'sl_SI'}

    if text == '':
        st.markdown("<h4 style='color:orange;'>CLICK EXTRACT AND TRANSLATE</h4>", unsafe_allow_html=True) 
    else: 
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        encoded_text = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id[lang_dict[lang]])
        out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        st.markdown("<h2 style='color:orange;'>TRANSLATED TEXT:</h2>", unsafe_allow_html=True)
        st.write('', str(out).strip('][\''))
        
lang = st.selectbox(" ",('Arabic', 'Czech', 'German', 'English', 'Spanish', 'Estonian', 'Finnish', 'French', 'Gujarati', 'Hindi', 'Italian', 'Japanese', 'Kazakh', 'Korean', 'Lithuanian', 'Latvian', 'Burmese', 'Nepali', 'Dutch', 'Romanian', 'Russian', 'Sinhala', 'Turkish', 'Vietnamese', 'Chinese', 'Afrikaans', 'Azerbaijani', 'Bengali', 'Persian', 'Hebrew', 'Croatian', 'Indonesian', 'Georgian', 'Khmer', 'Macedonian', 'Malayalam', 'Mongolian', 'Marathi', 'Polish', 'Pashto', 'Portuguese', 'Swedish', 'Swahili', 'Tamil', 'Telugu', 'Thai', 'Tagalog', 'Ukrainian', 'Urdu', 'Xhosa', 'Galician', 'Slovene'))

if inp_tran is not None:
    Translate(inp_tran, lang)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)