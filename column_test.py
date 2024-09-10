import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from qdrant_client import QdrantClient
import requests
import json

# Set page config
st.set_page_config(page_title="AI Chat with Qdrant Search", layout="wide")

# Set up OpenAI and Qdrant clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
qdrant_client = QdrantClient(url=st.secrets["qdrant_url"], port=443, api_key=st.secrets["qdrant_api_key"])

# Constants
GPT_MODEL = "gpt-4o"  # Make sure this is the correct model name
EMBEDDING_MODEL = "text-embedding-3-large"

# Function to generate embeddings
def generate_embeddings(text):
    try:
        response = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

# Function to search Qdrant
def search_collection(qdrant_client, collection_name, user_query_embedding, limit=5):
    try:
        response = qdrant_client.search(
            collection_name=collection_name,
            query_vector=user_query_embedding,
            limit=limit,
            with_payload=True
        )
        return response
    except Exception as e:
        st.error(f"Error searching Qdrant collection: {str(e)}")
        return []

# Tool call function
def search_qdrant(user_input: str, collection_name: str, limit: int = 5):
    user_query_embedding = generate_embeddings(user_input)
    if user_query_embedding is None:
        return []

    results = search_collection(qdrant_client, collection_name, user_query_embedding, limit)
    formatted_results = []
    for result in results:
        formatted_results.append({
            "score": result.score,
            "payload": result.payload
        })

    return formatted_results

# Function to handle AI response generation
def generate_ai_response(messages):
    try:
        completion = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        return completion
    except Exception as e:
        st.error(f"Error generating AI response: {str(e)}")
        return None

# Function to extract letter from response
def extract_letter(response):
    letter_start = response.find("<letter>")
    letter_end = response.find("</letter>")
    if letter_start != -1 and letter_end != -1:
        return response[letter_start:letter_end + 9]
    return None

# Streamlit app
st.title("AI Chat with Qdrant Search")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'last_letter' not in st.session_state:
    st.session_state['last_letter'] = ""

if 'tool_calls' not in st.session_state:
    st.session_state['tool_calls'] = []

# Generate HTML for independent scrolling
scrollable_messages = "<br>".join([f"<p>{message['role']}: {message['content']}</p>" for message in st.session_state['messages'] if message["role"] != "function"])
last_letter = st.session_state['last_letter'] if st.session_state['last_letter'] else "No draft letter to display yet."

# Define custom HTML for independent scrolling columns
html_code = f"""
    <style>
        .container {{
            display: flex;
        }}
        .scrollable {{
            height: 500px;
            overflow-y: scroll;
            flex: 2;
            padding: 20px;
            border-right: 2px solid #f0f0f0;
        }}
        .static {{
            flex: 1;
            padding: 20px;
            position: sticky;
            top: 0;
        }}
    </style>

    <div class="container">
        <div class="scrollable">
            <h2>Chat</h2>
            {scrollable_messages}
        </div>
        <div class="static">
            <h2>Last Draft Letter</h2>
            <p>{last_letter}</p>
        </div>
    </div>
"""

# Render the custom HTML
components.html(html_code, height=600)

# Chat input at the bottom
user_input = st.text_input("Type your message here:")

# Handle user input and generate AI response
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate AI response
    messages = [
        {"role": "system", "content": "Du är en hjälpsam AI-assistent för Falkenbergs kommun."}
    ] + st.session_state.messages

    completion = generate_ai_response(messages)
    if completion:
        response_message = completion.choices[0].message

        # Handle function calls
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "search_qdrant":
                    function_args = safe_json_loads(tool_call.function.arguments)
                    search_results = search_qdrant(**function_args)
                    st.session_state.messages.append({
                        "role": "function",
                        "name": "search_qdrant",
                        "content": json.dumps(search_results)
                    })
                    st.session_state['tool_calls'].append({
                        "function": "search_qdrant",
                        "arguments": function_args,
                        "results": search_results
                    })

            completion = generate_ai_response(st.session_state.messages)
            if completion:
                response_message = completion.choices[0].message

        full_response = response_message.content
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Extract letter if present
        letter = extract_letter(full_response)
        if letter:
            st.session_state['last_letter'] = letter

    st.rerun()

# Button to clear the chat history
if st.button("Clear Chat History"):
    st.session_state['messages'] = []
    st.session_state['last_letter'] = ""
    st.session_state['tool_calls'] = []
    st.rerun()




# import streamlit as st
# import streamlit.components.v1 as components

# # Generate content for the scrollable section
# scrollable_content = "<br>".join([f"Scrollable content {i}" for i in range(1, 51)])

# # Content for the static section
# static_content = "This is the static column that remains in place as you scroll."

# # Define custom HTML for independent scrolling columns using f-string
# html_code = f"""
#     <style>
#         .container {{
#             display: flex;
#         }}
#         .scrollable {{
#             height: 500px;
#             overflow-y: scroll;
#             flex: 2;
#             padding: 20px;
#             border-right: 2px solid #f0f0f0;
#         }}
#         .static {{
#             flex: 1;
#             padding: 20px;
#             position: sticky;
#             top: 0;
#         }}
#     </style>

#     <div class="container">
#         <div class="scrollable">
#             <h2>Scrollable Column</h2>
#             <p>{scrollable_content}</p>
#         </div>
#         <div class="static">
#             <h2>Static Column</h2>
#             <p>{static_content}</p>
#         </div>
#     </div>
# """

# # Render the custom HTML
# components.html(html_code, height=600)
