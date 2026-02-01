import streamlit as st
from nltk.chat.util import Chat, reflections

pairs = [
    [
        r"(.*)my name is (.*)",
        ["Hello %2, How are you today ?"]
    ],
    [
        r"(hi|hey|hello|hola|holla)(.*)",
        ["Hello", "Hey there"]
    ],
    [
        r"(.*)help(.*)",
        ["I can help you"]
    ],
    [
        r"(.*) your name ?",
        ["My name is thecleverprogrammer, but you can just call me robot and I'm a chatbot."]
    ],
    [
        r"how are you(.*)",
        ["I'm doing very well", "I am great!"]
    ],
    [
        r"sorry(.*)",
        ["It's alright", "It's OK, never mind that"]
    ],
    [
        r"i'm (.*) (good|well|okay|ok)",
        ["Nice to hear that", "Alright, great!"]
    ],
    [
        r"(.*)created(.*)",
        ["Prakash created me using Python's NLTK library", "Top secret ;)"]
    ],
    [
        r"(.*)(location|city)(.*)",
        ["Hyderabad, India"]
    ],
    [
        r"(.*)raining in (.*)",
        ["No rain in the past 4 days here in %2", "In %2 there is a 50% chance of rain"]
    ],
    [
        r"(.*)(sports|game|sport)(.*)",
        ["I'm a very big fan of Cricket"]
    ],
    [
        r"who (.*)(cricketer|batsman)",
        ["Virat Kohli"]
    ],

    # ✅ NLP RULE (FIXED)
    [
        r"(what is|explain|define)(.*)nlp(.*)",
        ["NLP stands for Natural Language Processing. Natural Language Processing is a branch of artificial intelligence (AI) that focuses on enabling computers to understand, interpret, generate, and interact with human language (text and speech)"]

    ],

    [
        r"quit",
        ["Bye for now. See you soon :)", "It was nice talking to you. See you soon :)"]
    ],

    # ✅ DEFAULT RULE — ALWAYS LAST
    [
        r"(.*)",
        ["Our customer service will reach you"]
    ],
]

chatbot = Chat(pairs, reflections)

# Streamlit UI
st.set_page_config(page_title="NLTK Chatbot", page_icon="🤖")
st.title("🤖 NLTK Chatbot")
st.write("Type lowercase English language. Type **quit** to exit.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Say something...")

if user_input:
    user_input = user_input.lower()  # ✅ IMPORTANT

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response = chatbot.respond(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
