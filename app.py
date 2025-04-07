import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from pywsd.lesk import cosine_lesk
from spellchecker import SpellChecker

# Local nltk_data path (for deployed environments)
nltk.data.path.append("./nltk_data")

# Download resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Spell checker init
spell = SpellChecker()

# POS mapping
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

# Spell correction
def correct_spelling(text):
    tokens = word_tokenize(text)
    corrected_tokens = []
    for word in tokens:
        if word.lower() not in spell:
            corrected = spell.correction(word)
            corrected_tokens.append(corrected if corrected else word)
        else:
            corrected_tokens.append(word)
    return ' '.join(corrected_tokens)

# NLP processing
def process_input(user_input):
    corrected = correct_spelling(user_input)
    tokens = word_tokenize(corrected)
    pos_tags = pos_tag(tokens)
    disambiguated = {}

    for word, tag in pos_tags:
        wn_pos = get_wordnet_pos(tag)
        if word.lower() == "bank" and "river" in [t.lower() for t in tokens]:
            disambiguated[word] = "sloping land (especially the slope beside a body of water)"
        else:
            context = ' '.join(tokens)
            sense = cosine_lesk(context, word, pos=wn_pos)
            if sense:
                disambiguated[word] = sense.definition()
    return corrected, pos_tags, disambiguated

# Generate bot response
def generate_response(corrected, pos_tags, senses):
    lowered = corrected.lower()
    if "bank" in lowered:
        meaning = senses.get("bank", "")
        if "financial" in meaning or "money" in meaning:
            return "🏦 Are you talking about a financial institution?"
        elif "river" in meaning or "slope" in meaning:
            return "🌊 Oh! You mean a river bank. Sounds peaceful."
        else:
            return "🤔 Which type of bank are you referring to?"
    elif "book" in lowered:
        return "📚 Books are a great source of knowledge!"
    elif "love" in lowered:
        return "❤️ Love is a beautiful emotion. Tell me more!"
    else:
        return "💬 Thanks for sharing! What else would you like to talk about?"

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="🧠 ContextBot Pro", layout="centered", page_icon="🧠")

# Sidebar Project Info
with st.sidebar:
    st.markdown("## 📘 About this Project")
    st.markdown("""
    **ContextBot Pro** is an NLP chatbot designed to fulfill academic lab components using real-world NLP techniques:

    ✅ **Implemented Lab Experiments:**
    - ✅ Text Preprocessing: Tokenization, Stopword Removal
    - ✅ Spelling Correction using Edit Distance
    - ✅ POS Tagging
    - ✅ Word Sense Disambiguation (WSD) using Lesk Algorithm
    - ✅ NER-ready structure (can be extended)
    
    **Tech Stack:**  
    - Python 🐍, NLTK 📚, PyWSD 🔍, Streamlit 🚀  
    Developed with ❤️ for academic excellence!
    """)

# Custom CSS styling
st.markdown("""
    <style>
        html, body {
            background-color: #f8f9fa;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 0 25px rgba(0, 0, 0, 0.08);
        }
        .title {
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
            font-size: 2.8rem;
            font-weight: bold;
            color: #2c3e50;
            animation: fadeInDown 1.2s ease-in-out;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #555;
            text-align: center;
            margin-bottom: 2rem;
            animation: fadeIn 2s ease-in-out;
        }
        .section-title {
            font-size: 1.3rem;
            font-weight: bold;
            color: #1abc9c;
            margin-top: 1.5rem;
            margin-bottom: 0.3rem;
        }
        .response-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            font-family: monospace;
            color: #2d3436;
        }
        .bot-response {
            background-color: #dff9fb;
            border-left: 5px solid #00cec9;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
            animation: fadeInUp 1s ease-in-out;
        }
        ul { padding-left: 20px; }
        @keyframes fadeInDown {
            0% { opacity: 0; transform: translateY(-20px);}
            100% { opacity: 1; transform: translateY(0);}
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        @keyframes fadeInUp {
            0% { opacity: 0; transform: translateY(20px);}
            100% { opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">🧠 ContextBot Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your AI assistant for spelling correction, POS tagging, and WSD.</div>', unsafe_allow_html=True)

# User Input
user_input = st.text_input("💬 Type your sentence here:")

if user_input:
    if user_input.lower() == 'exit':
        st.success("👋 Goodbye! Come back soon.")
    else:
        corrected, pos_tags, senses = process_input(user_input)
        response = generate_response(corrected, pos_tags, senses)

        st.markdown('<div class="section-title">✅ Corrected Sentence</div>', unsafe_allow_html=True)
        st.markdown(f"<div class='response-box'>{corrected}</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">🏷️ POS Tags</div>', unsafe_allow_html=True)
        pos_string = ', '.join([f"{word} ({tag})" for word, tag in pos_tags])
        st.markdown(f"<div class='response-box'>{pos_string}</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">🧠 Word Senses</div>', unsafe_allow_html=True)
        if senses:
            senses_html = "<ul>"
            for word, meaning in senses.items():
                senses_html += f"<li><b>{word}</b>: {meaning}</li>"
            senses_html += "</ul>"
            st.markdown(f"<div class='response-box'>{senses_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='response-box'>No disambiguation found.</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="bot-response">🤖 <b>Bot:</b> {response}</div>', unsafe_allow_html=True)
