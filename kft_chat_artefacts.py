import streamlit as st
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
    print(collection_name)
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
    print('search qdrant', user_input)
    print('\n\n')
    print('collection name', collection_name)
    user_query_embedding = generate_embeddings(user_input)
    print('embeddings done')
    if user_query_embedding is None:
        return []

    results = search_collection(qdrant_client, collection_name, user_query_embedding, limit)
    print('search collection done', results)
    formatted_results = []
    for result in results:
        formatted_results.append({
            "score": result.score,
            "payload": result.payload
        })

    return formatted_results

# Set up tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_qdrant",
            "description": "Search the Qdrant database for similar entries based on the user's input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "The user's search query or input text."
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "The name of the Qdrant collection to search in.",
                        "enum": ["FalkenbergsKommunsHemsida_1000char_chunks", "mediawiki"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "The number of similar results to return.",
                        "default": 5
                    }
                },
                "required": ["user_input", "collection_name"]
            }
        }
    }
]

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

# Function to safely parse JSON
def safe_json_loads(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        st.warning("Error parsing JSON from function arguments. Using empty dict.")
        return {}

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

# Sidebar for tool call information
with st.sidebar:
    st.subheader("Tool Call Information")
    if st.session_state['tool_calls']:
        for call in st.session_state['tool_calls']:
            st.write(f"Function: {call['function']}")
            st.write(f"Arguments: {call['arguments']}")
            st.json(call['results'])
            st.write("---")
    else:
        st.write("No tool calls made yet.")

# Create two columns
col1, col2 = st.columns(2)

# Column 1: Chat interface with scrolling
with col1:
    st.subheader("Chat")
    chat_container = st.container()

    # Wrap the chat content in a scrollable div
    st.markdown('<div class="scrollable-column">', unsafe_allow_html=True)

    for message in st.session_state.messages:
        if message["role"] != "function":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Close the scrollable div
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input at the bottom of col1
    user_input = st.chat_input("Type your message here:")

# Column 2: Display last letter (fixed)
with col2:
    with st.container():
        st.subheader("Last Draft Letter")
        letter_container = st.empty()
        if st.session_state['last_letter']:
            letter_container.markdown(st.session_state['last_letter'])
        else:
            letter_container.write("No draft letter to display yet.")

# Handle user input and generate AI response
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate AI response
    messages = [
        {"role": "system", "content": "Du är en hjälpsam AI-assistent för Falkenbergs kommun. Använd search_qdrant-funktionen när du behöver hitta information. När du skriver ett utkast till svar för KFT-handläggaren som svar till en invånare, omslut det med <letter></letter>-taggar."}
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

            # Get the final response after function calls
            completion = generate_ai_response(st.session_state.messages)
            if completion:
                response_message = completion.choices[0].message

        # Display the final response
        full_response = response_message.content
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Extract letter if present
        letter = extract_letter(full_response)
        if letter:
            st.session_state['last_letter'] = letter
            letter_container.markdown(letter)

    # Rerun the app to update the chat history and letter
    st.rerun()

# Add a button to clear the chat history
if st.button("Clear Chat History"):
    st.session_state['messages'] = []
    st.session_state['last_letter'] = ""
    st.session_state['tool_calls'] = []
    st.rerun()
