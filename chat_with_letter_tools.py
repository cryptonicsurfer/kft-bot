import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
import re
import json

# Set page config
st.set_page_config(layout="wide")

# Set the title of the Streamlit app
st.title("KFT utkastgenereraren")

# Load OpenAI API key from Streamlit secrets
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
qdrant_client = QdrantClient(url=st.secrets["qdrant_url"], port=443, api_key=st.secrets["qdrant_api_key"])


# Define the GPT model to be used
GPT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"

# Function to safely parse JSON
def safe_json_loads(json_string):
    # Split the JSON string if multiple JSON objects are concatenated
    json_objects = re.findall(r'\{.*?\}(?=\{|\Z)', json_string)
    all_arguments = []

    for obj in json_objects:
        try:
            parsed = json.loads(obj)
            all_arguments.append(parsed)
        except json.JSONDecodeError:
            st.warning(f"Error parsing JSON from function arguments: {obj}. Using empty dict.")
            all_arguments.append({})

    return all_arguments

# Function to generate embeddings
def generate_embeddings(text):
    try:
        response = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

# Function to search Qdrant
def search_collection(qdrant_client, collection_name, user_query_embedding, limit=3):
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
def search_qdrant(user_input: str='', limit: int = 3):
    if user_input == '': return ''
    print('Searching', user_input)
    user_query_embedding = generate_embeddings(user_input)
    if user_query_embedding is None:
        return []

    results = search_collection(qdrant_client, 'FalkenbergsKommunsHemsida_1000char_chunks', user_query_embedding, limit)
    results2 = search_collection(qdrant_client, 'mediawiki', user_query_embedding, limit)
    formatted_results = []
    for result in results:
        formatted_results.append({
            "score": result.score,
            "payload": result.payload
        })
    for result in results2:
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
            "description": "Search the falkenbergs kommuns databses/collections for policies and procedures. Use this when you need to find additional information to support the case worker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "Full context of the entire case including all possible keywords. 5 to 20 words of context",
                        "example": "lekplats grönområde farligt barnlek trafikfara skötsel vägmärkesförordningen farthinder"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "The number of similar results to return.",
                        "default": 3
                    }
                },
                "required": ["user_input"]
            }
        }
    }
]

# Initialize session state for storing chat messages if not already set
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'letters' not in st.session_state:
    st.session_state['letters'] = ['']
if 'letter_placeholder' not in st.session_state:
    st.session_state['letter_placeholder'] = ''
if 'current_tool_call' not in st.session_state:
    st.session_state['current_tool_call'] = {'name': None, 'arguments': ''}




SYSTEM_MESSAGE = {
    "role": "system",
    "content": "Du är en hjälpsam assistent som hjälper en kommunanställd att författa ett svar till en invånare. Givet invånarfrågan, sammanställ relevant fakta på ett lättläst sätt, samt ge ett utkast på hur ett svar skulle kunna se ut. Ditt svar riktas till en anställd på kommunen och ska utgöra ett stöd för den anställde att återkoppla direkt till den som ställer frågan. Om du har rätt fakta för att ge ett korrekt svar, skriv det. Om inte, skriv att kommunen har tagit emot synpunkten och diariefört den men att det inte är säkert att det finns resurser att prioritera just denna fråga. Inkludera alltid källor. Svara vänligt men kortfattat. Svaret börjar med: 'Hej Namn,' och avslutas med: 'Med vänliga hälsningar, [Namn], [Avdelning på kommunen]'. Svaret ska formateras i markdown och markeras inom tags <letter>[letter content in markdown]</letter>, efter closing tag lista länk till källorna som du har baserat ditt svar på. Svaret ska aldrig hänvisa tillbaka till en specifik person, hänvisa om nödvändigt till kontaktcenter  Tel: 0346-88 60 00 Mejl: kontaktcenter@falkenberg.se."
}

cola, colb = st.columns(2)

user_input = st.chat_input("Skriv medborgarfråga eller instruktioner här ...")
with cola:
    with st.container(border=True, height=600):
        # Display previous chat messages
        for message in st.session_state.messages:
            if message["role"] == "function":
                continue
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
                completion = openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[SYSTEM_MESSAGE] + [
                        {"role": m["role"],
                        "content": m["content"],
                        **({"name": m["name"]} if m["role"] == "function" else {})
                        }
                        for m in st.session_state.messages
                    ],
                    stream=True,
                    tools=tools,
                    temperature=0.2,
                    tool_choice="auto",
                )


                # Handle text completions
                for chunk in completion:
                    # Handle tool call
                    choice = chunk.choices[0]
                    if choice.delta.tool_calls:

                        tool_call = choice.delta.tool_calls[0]

                        # Accumulate tool call arguments
                        if tool_call.function.arguments is not None:
                            st.session_state['current_tool_call']['arguments'] += tool_call.function.arguments

                        # Capture the tool call function name
                        if tool_call.function.name is not None:
                            st.session_state['current_tool_call']['name'] = tool_call.function.name

                        # Continue to accumulate tool call arguments until all parts are received
                        continue

                    # When the tool call is complete, execute the tool function
                    if choice.finish_reason == "tool_calls" and st.session_state['current_tool_call']['name']:
                        function_name = st.session_state['current_tool_call']['name']
                        function_args_list = safe_json_loads(st.session_state['current_tool_call']['arguments'])

                        for function_args in function_args_list:
                            # Perform the tool function based on the function name
                            if function_name == "search_qdrant":
                                search_results = search_qdrant(**function_args)
                                st.session_state.messages.append({
                                    "role": "function",
                                    "name": "search_qdrant",
                                    "content": json.dumps(search_results)
                                })

                        # Reset tool call state for future calls
                        st.session_state['current_tool_call'] = {'name': None, 'arguments': ''}
                        # Call openai and give it the function output
                        completion = openai_client.chat.completions.create(
                            model=GPT_MODEL,
                            messages=[SYSTEM_MESSAGE] + [
                                {"role": m["role"],
                                "content": m["content"],
                                **({"name": m["name"]} if m["role"] == "function" else {})
                                }

                                for m in st.session_state.messages
                            ],
                            stream=True,
                            tools=tools,
                            temperature=0.2,
                            tool_choice="auto",
                        )
                        for chunk in completion:
                            choice = chunk.choices[0]
                            if choice.finish_reason == "stop":
                                message_placeholder.markdown(message_response)
                                if st.session_state.letter_placeholder != '':
                                    st.session_state.letters.append(st.session_state.letter_placeholder)
                                st.session_state.letter_placeholder = ''
                                break

                            if choice.delta.content:
                                full_response += choice.delta.content

                                if '<letter>' in full_response and '</letter>' not in full_response:
                                    message_response = message_response.replace('<letter','Skriver brev...')
                                    st.session_state.letter_placeholder += choice.delta.content
                                else:
                                    message_response += choice.delta.content
                                message_placeholder.markdown(message_response + "▌")
                        break

                    if choice.finish_reason == "stop":
                        message_placeholder.markdown(message_response)
                        if st.session_state.letter_placeholder != '':
                            st.session_state.letters.append(st.session_state.letter_placeholder)
                        st.session_state.letter_placeholder = ''
                        break

                    if choice.delta.content:
                        full_response += choice.delta.content

                        if '<letter>' in full_response and '</letter>' not in full_response:
                            message_response = message_response.replace('<letter','Skriver brev...')
                            st.session_state.letter_placeholder += choice.delta.content
                        else:
                            message_response += choice.delta.content
                        message_placeholder.markdown(message_response + "▌")

            # Add bot's reply to session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    with colb:
        with st.container(border=True, height=600):
            if st.session_state.letter_placeholder:
                letter_content = st.session_state.letter_placeholder.replace('</letter', '').replace('>','')
            elif st.session_state.letters:
                letter_content = st.session_state.letters[-1].replace('</letter', '').replace('>','')
            else:
                letter_content = ""

            st.code(letter_content, wrap_lines=True)
