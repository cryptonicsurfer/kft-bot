import streamlit as st

set_page_config = st.set_page_config(layout= "wide")

# Create custom CSS
css = """
<style>
    [data-testid="column"] {
        background-color: #e6f3ff;  /* This is now the padding color */
        border: 1px solid #007bff;
        border-radius: 5px;
        padding: 10px;
    }
    [data-testid="column"] > div {
        background-color: #ffffff;  /* This is the fill color for the content area */
        padding: 10px;
        border-radius: 3px;
    }
</style>
"""

# Inject custom CSS with st.markdown
st.markdown(css, unsafe_allow_html=True)




if "messages" not in st.session_state:
    st.session_state.messages = []

cola, colb = st.columns(2)
with cola:
    with st.container(border=True, height=800):
        col1, col2 = st.columns(2)

        with col1:
            user_input=st.chat_input("Say something")
            # add user_input to session state
            if user_input:
                st.session_state.messages.append(user_input)

        with col2:
            if user_input:
                # display all messages in session state, ie history so iterate over list

                for message in st.session_state.messages:
                    st.chat_message(user_input)
            else:
                st.write("No input from user yet")

with colb:

    with st.container(border=True, height=800):
        st.write("This is outside the if block")
