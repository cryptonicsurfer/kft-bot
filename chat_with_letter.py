import streamlit as st
from openai import OpenAI
import re

SYSTEM_MESSAGE = {"role": "system", 
                  "content": "Du är en hjälpsam assistent som ibland svarar med ett separat brev markerat inom tags <letter>[letter content in markdown]</letter>"}

# Set page config
st.set_page_config(layout="wide")

# Set the title of the Streamlit app
st.title("Simple Artefact chat")

# Load OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Define the GPT model to be used
GPT_MODEL = "gpt-4o-mini"

# Initialize session state for storing chat messages if not already set
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'letters' not in st.session_state:
    st.session_state['letters'] = ['']
if 'letter_placeholder' not in st.session_state:
    st.session_state['letter_placeholder'] = ''

cola, colb = st.columns(2)

user_input = st.chat_input("Svara ...")
with cola:
    with st.container(border=True, height=600):
        # Display previous chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # st.markdown(message["content"])
                if '<letter>' in message["content"] and '</letter>' in message["content"]:
                    # Split the content into parts
                    parts = re.split(r'(<letter>.*?</letter>)', message["content"], flags=re.DOTALL)
                    
                    # Display content before the letter
                    if parts[0].strip():
                        st.markdown(parts[0].strip())
                    
                    # Extract and display the letter content
                    letter_content = re.search(r'<letter>(.*?)</letter>', parts[1], re.DOTALL)
                    if letter_content:
                        with st.expander("Brev"):
                            st.markdown(letter_content.group(1))
                    
                    # Display content after the letter
                    if len(parts) > 2 and parts[2].strip():
                        st.markdown(parts[2].strip())
                else:
                    # If there's no letter tag, display the content as is
                    st.markdown(message["content"])

        
        if user_input:
            # Add user's message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Stream the GPT-4 reply
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                message_response = ""
                completion = client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[SYSTEM_MESSAGE] + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True
                )
                for chunk in completion:
                    if chunk.choices[0].finish_reason == "stop": 
                        message_placeholder.markdown(message_response)
                        if st.session_state.letter_placeholder != '': 
                            st.session_state.letters.append(st.session_state.letter_placeholder)
                        st.session_state.letter_placeholder = ''
                        break

                    full_response += chunk.choices[0].delta.content
                    
                    if '<letter>' in full_response and '</letter>' not in full_response:
                        message_response = message_response.replace('<letter','Skriver brev...')
                        st.session_state.letter_placeholder += chunk.choices[0].delta.content
                    else:
                        message_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(message_response + "▌")

            # Add bot's reply to session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})

with colb:
    with st.container(border=True, height=600):
        st.write(st.session_state['letter_placeholder'].replace('</letter',''))
        st.write(st.session_state.letters[-1].replace('</letter',''))