import streamlit as st
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer


# Set environment variable for TOKENIZERS_PARALLELISM
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
openai_api_key = st.secrets['OPENAI_API_KEY'] #os.getenv('OPENAI_API_KEY')
qdrant_api_key = st.secrets['qdrant_api_key'] #os.getenv('qdrant_api_key')
qdrant_url = "https://qdrant.utvecklingfalkenberg.se"
collection_name = "KFT_knowledge_base_KBLabSwedishEmbeddings"

# Initialize clients
openai_client = OpenAI(api_key=openai_api_key)
qdrant_client = QdrantClient(url=qdrant_url, port=443, https=True, api_key=qdrant_api_key)

# Initialize SentenceTransformer model
model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')

def generate_embeddings(text):
    return model.encode(text).tolist()

def search_collection(qdrant_client, collection_name, user_query_embedding):
    response = qdrant_client.search(
        collection_name=collection_name,
        query_vector=user_query_embedding,
        limit=3,  # Adjust based on needs
        with_payload=True,
        score_threshold = 0.6
    )
    # Directly return the response or the relevant part of it
    return response  # Adjust this line if the structure is different

# Stream the OpenAI chat response
def get_chat_response_streaming(user_message, instructions_prompt, model="gpt-4-turbo-preview", client=None):
    if client is None:
        client = OpenAI(api_key=openai_api_key)
    full_response = ""
    message_placeholder = st.empty()  # Placeholder for displaying the response
    
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": instructions_prompt}],
        stream=True
    )
    
    for chunk in completion:
        if chunk.choices[0].finish_reason == "stop":
            message_placeholder.markdown(full_response)  # Display the final response
            # break  # Exit the loop since the response is complete
            return full_response  # Return the final response
        full_response += chunk.choices[0].delta.content
        message_placeholder.markdown(full_response + "▌")


st.title("Demo KFT - utkastsgenererare")
with st.form(key='user_query_form', clear_on_submit=True):
    user_input = st.text_input("Klistra in fråga/klagomål från invånare här:", key="user_input")
    st.caption("Svaren genereras av en AI-bot, som kan begå misstag. Frågor och svar lagras i utvecklingssyfte. Skriv inte personuppgifter i fältet.")
    submit_button = st.form_submit_button("Sök")

if submit_button and user_input:
    user_embedding = generate_embeddings(user_input)
    search_results = search_collection(qdrant_client, collection_name, user_embedding)
    
    similar_texts = [
        (result.payload['text'], result.payload['file_source'], result.score)
        for result in search_results
    ]
    
    if similar_texts:  # Ensure there are results
        with st.expander("Se relevanta källor"):
            for index, (text, source, score) in enumerate(similar_texts):
                st.write(f"Resultat {index + 1}:")
                st.write("Källa:", source)  # Display the source
                st.write("Träffsäkerhet:", score)  # Display the score
                st.write(f"Text från dokument:\n {text}")#, text)  # Display the text
                st.write("---")  # Optional: add a separator line for better readability


    instructions_prompt = f'Givet denna fråga: {user_input} och kontexten från en databas: {search_results}, sammanställ relevant fakta på ett lättläst sätt, samt ge ett utkast på hur ett svar skulle kunna se ut. Ditt svar riktas till en anställd på kommunen och skall utgöra ett stöd för den antsällde att återkoppla direkt till den som ställer frågan. Innehåller {user_input} både en fråga och synpunkt eller klagomål, addreserar du båda utifrån din fakta. Om du har fått rätt kontext i form av fakta för att ge ett korrekt svar så skriver du det, om inte så skriver du att kommunen har tagit emot synpunkten och diariefört den men att det inte är säkert att det finns resurser att prioritera just denna fråga. Inkludera källa för ditt svar. Svara vänligt men kortfattat.'
    
    # Stream or display the GPT-4 reply based on instructions_prompt
    answer = get_chat_response_streaming(user_input, instructions_prompt)
    # st.code(answer)

