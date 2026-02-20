import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os
import speech_recognition as sr
from fpdf import FPDF
from gtts import gTTS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

st.set_page_config(page_title="NyayaSahayak", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
.chat-container { border-radius: 10px; padding: 20px; background-color: #f0f2f6; margin-bottom: 20px; }
.bot-msg { color: #0e1117; font-weight: bold; }
.user-msg { color: #0068c9; font-weight: bold; text-align: right; }
.stButton>button { width: 100%; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_legal_chain():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever


def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        filename = "response.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None


def generate_pdf(complaint_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="FORMAL LEGAL COMPLAINT", ln=1, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=complaint_text)

    filename = "complaint.pdf"
    pdf.output(filename)
    return filename


def recognize_speech():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Listening... Speak your issue now!")

        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            text = r.recognize_google(audio)
            return text

        except sr.UnknownValueError:
            st.error("Could not understand audio. Try speaking clearer.")
            return None

        except Exception as e:
            st.error(f"Microphone error: {e}")
            return None


st.title("⚖️ NyayaSahayak (Justice Helper)")
st.subheader("Your Multilingual AI Legal Assistant")

col1, col2 = st.columns([1, 2])

with col1:
    # st.image("", width=150)
    st.markdown("### How to use:")
    st.markdown("1. Choose Text or Voice input.")
    st.markdown("2. Ask your question.")
    st.markdown("3. Get a simplified legal answer + audio playback.")

with col2:
    user_query = ""
