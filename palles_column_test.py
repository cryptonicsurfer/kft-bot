import streamlit as st
from openai import OpenAI
import re

# Set page config
st.set_page_config(layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "display_mode" not in st.session_state:
    st.session_state.display_mode = "chat"

# Function to get chat response (streaming)
def get_chat_response_streaming(user_message, instructions_prompt, model="gpt-3.5-turbo", client=None):
    if client is None:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    messages = [
        {"role": "system", "content": instructions_prompt},
        {"role": "user", "content": user_message}
    ]
    
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)
    return stream

# Function to extract letter content
def extract_letter_content(text):
    pattern = r'<letter>(.*?)</letter>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

# Function to handle new messages
def handle_new_message(user_input):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in get_chat_response_streaming(user_input, "You are a helpful assistant."):
            full_response += chunk.choices[0].delta.content or ""
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Check for letter tags
    letter_content = extract_letter_content(full_response)
    if letter_content:
        st.session_state.display_mode = "split"
        st.session_state.letter_content = letter_content
        st.rerun()

# Main app layout
def main():
    if st.session_state.display_mode == "chat":
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.title("Chat Interface")
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            user_input = st.chat_input("Type your message here...")
            
            if user_input:
                handle_new_message(user_input)
    
    elif st.session_state.display_mode == "split":
        cola, colb = st.columns(2)
        
        with cola:
            st.title("Chat History")
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input in split view
            user_input = st.chat_input("Type your message here...")
            
            if user_input:
                handle_new_message(user_input)
        
        with colb:
            st.title("Letter Content")
            for content in st.session_state.letter_content:
                st.markdown(content)
        
        if st.button("Return to Chat"):
            st.session_state.display_mode = "chat"
            st.rerun()

if __name__ == "__main__":
    main()