import streamlit as st
from gtts import gTTS
import tempfile
from langdetect import detect, lang_detect_exception
from deep_translator import GoogleTranslator
import pycountry
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import nltk

# ----------------- NLTK SETUP -----------------
nltk.download('punkt')
nltk.download('words')

# ----------------- FUNCTIONS -----------------
def read_aloud(text, language='en'):
    tts = gTTS(text=text, lang=language)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        st.audio(open(f.name, "rb").read(), format="audio/mp3")

def generate_wordcloud(text):
    english_words = set(nltk.corpus.words.words())
    words = word_tokenize(text.lower())
    valid_words = [w for w in words if w.isalpha() and w in english_words]

    if not valid_words:
        raise ValueError("Not enough valid English words")

    wc = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(" ".join(valid_words))

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

def get_language_code(name):
    try:
        return pycountry.languages.lookup(name).alpha_2
    except:
        return None

# ----------------- STREAMLIT APP -----------------
st.set_page_config(page_title="Globalize", layout="wide")
st.title("🌍 Globalize")

col1, col2 = st.columns(2)

with col1:
    paragraph = st.text_area("Enter one paragraph:")

with col2:
    all_languages = sorted(
        [lang.name for lang in pycountry.languages if hasattr(lang, "alpha_2")]
    )
    target_languages_input = st.multiselect(
        "Select the desired languages for translation:",
        all_languages
    )

# ----------------- READ ALOUD -----------------
if st.button("Read Aloud"):
    if paragraph.strip():
        read_aloud(paragraph)
    else:
        st.warning("Please enter text")

# ----------------- LANGUAGE DETECTION -----------------
paragraph_language = "en"

if paragraph.strip():
    try:
        paragraph_language = detect(paragraph)
        language_name = pycountry.languages.get(alpha_2=paragraph_language).name
        st.success(f"Detected language: {language_name}")
    except lang_detect_exception.LangDetectException:
        st.warning("Could not detect language")

# ----------------- TRANSLATE TO ENGLISH -----------------
if paragraph_language != 'en':
    translated_paragraph = GoogleTranslator(
        source='auto',
        target='en'
    ).translate(paragraph)

    st.subheader("Translated to universal language English:")
    st.write(translated_paragraph)
else:
    translated_paragraph = paragraph

# ----------------- WORD CLOUD -----------------
if translated_paragraph.strip():
    try:
        st.sidebar.subheader("☁️ Word Cloud")
        fig = generate_wordcloud(translated_paragraph)
        st.sidebar.pyplot(fig)
    except:
        st.sidebar.warning("Word cloud needs more meaningful English words")

# ----------------- TRANSLATE & READ ALOUD -----------------
if st.button("Translate and Read Aloud"):
    for lang_name in target_languages_input:
        lang_code = get_language_code(lang_name)
        if not lang_code:
            continue

        try:
            translated_text = GoogleTranslator(
                source='auto',
                target=lang_code
            ).translate(paragraph)

            st.subheader(f"Translated paragraph in {lang_name}:")
            st.write(translated_text)

            read_aloud(translated_text, lang_code)

        except Exception as e:
            st.error(f"Translation to {lang_name} failed")
