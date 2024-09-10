import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
import requests
import json

# Set page config
st.set_page_config(page_title="AI Chat with Qdrant Search", layout="wide")

# Set up OpenAI and Qdrant clients
openai_client = OpenAI(api_key=st.secrets["openai_api_key"])
qdrant_client = QdrantClient(url=st.secrets["qdrant_url"], api_key=st.secrets["qdrant_api_key"])

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

# Column 1: Chat interface
with col1:
    st.subheader("Chat")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# Column 2: Display last letter
with col2:
    st.subheader("Last Draft Letter")
    if st.session_state['last_letter']:
        st.markdown(st.session_state['last_letter'])
    else:
        st.write("No draft letter to display yet.")

# Chat input at the bottom
with col1:
    user_input = st.chat_input("Type your message here:")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with chat_container.chat_message("user"):
            st.markdown(user_input)

        # Generate AI response
        with chat_container.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Prepare messages for the API call
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant for Falkenberg municipality. Use the search_qdrant function when you need to find information. When drafting a response for the KFT handläggare, enclose it in <letter></letter> tags."}
            ] + st.session_state.messages

            # Get the response
            completion = generate_ai_response(messages)
            if completion:
                response_message = completion.choices[0].message
                
                # Check for function calls
                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        if tool_call.function.name == "search_qdrant":
                            function_args = safe_json_loads(tool_call.function.arguments)
                            search_results = search_qdrant(**function_args)
                            print(search_results)
                            # Add the search results to the messages
                            st.session_state.messages.append({
                                "role": "function",
                                "name": "search_qdrant",
                                "content": json.dumps(search_results)
                            })

                            # Add tool call information to the sidebar
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
                message_placeholder.markdown(full_response)

                # Update chat history with AI response
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Extract letter if present
                letter = extract_letter(full_response)
                if letter:
                    st.session_state['last_letter'] = letter

        # Rerun the app to update the sidebar and chat history
        st.rerun()

# Add a button to clear the chat history
if st.button("Clear Chat History"):
    st.session_state['messages'] = []
    st.session_state['last_letter'] = ""
    st.session_state['tool_calls'] = []
    st.rerun()