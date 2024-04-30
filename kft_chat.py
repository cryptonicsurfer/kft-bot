import requests
import streamlit as st
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from streamlit_star_rating import st_star_rating

os.environ["TOKENIZERS_PARALLELISM"] = "false"

openai_api_key = st.secrets['OPENAI_API_KEY']
qdrant_api_key = st.secrets['qdrant_api_key']
qdrant_url = "https://qdrant.utvecklingfalkenberg.se"
collection_name = "KFT_knowledge_base_OpenAI_Large_chunk1000"
directus_api_url = "https://nav.utvecklingfalkenberg.se/items/kft_bot"
directus_params={"access_token":st.secrets['directus_token']}

openai_client = OpenAI(api_key=openai_api_key)
qdrant_client = QdrantClient(url=qdrant_url, port=443, https=True, api_key=qdrant_api_key)

def generate_embeddings(text):
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-large")
    return response.data[0].embedding

def search_collection(qdrant_client, collection_name, user_query_embedding):
    response = qdrant_client.search(collection_name=collection_name, query_vector=user_query_embedding, limit=5, with_payload=True) #score_threshold=0.4)
    return response

def get_chat_response_streaming(user_message, instructions_prompt, model="gpt-4-turbo-preview", client=None):
    if client is None:
        client = OpenAI(api_key=openai_api_key)
    full_response = ""
    message_placeholder = st.empty()
    completion = client.chat.completions.create(model=model, messages=[{"role": "system", "content": instructions_prompt}], stream=True)
    for chunk in completion:
        st.session_state['response_completed']=False
        if chunk.choices[0].finish_reason == "stop":
            st.session_state['response_completed']=True
            message_placeholder.markdown(full_response)
                # directus post    
            data={"prompt":user_message, "response": full_response}
            directus_response = requests.post(directus_api_url, json=data, params=directus_params)
            if directus_response.status_code == 200:
                response_data = directus_response.json()
                st.session_state['record_id'] = response_data['data']['id']  # Save the record ID for later update
                # st.success("Tack för din feedback!")
                print(st.session_state['record_id'])
            else:
                st.error("Något gick fel. Försök igen senare.")
            return full_response
        full_response += chunk.choices[0].delta.content
        message_placeholder.markdown(full_response + "▌")




def format_output(similar_texts, answer):
    text = f"Generated Response:\n{answer}\n\nSimilar Texts Found:\n"
    for index, (text_content, source, score) in enumerate(similar_texts):
        text += f"Resultat {index + 1} - Källa: {source}, Träffsäkerhet: {score}, Text: {text_content}\n"
    return text

st.title("Demo KFT - utkastsgenererare")
st.write(collection_name)
with st.form(key='user_query_form', clear_on_submit=True):
    user_input = st.text_input("Klistra in fråga/klagomål från invånare här:", key="user_input")
    st.caption("Svaren genereras av en AI-bot, som kan begå misstag. Frågor och svar lagras i utvecklingssyfte. Skriv inte personuppgifter i fältet.")
    submit_button = st.form_submit_button("Sök")

if submit_button and user_input:
    user_embedding = generate_embeddings(user_input)
    search_results = search_collection(qdrant_client, collection_name, user_embedding)
    similar_texts = [(result.payload['text'], result.payload['file_source'], result.score) for result in search_results]
    
    if similar_texts:
        with st.expander("Se relevanta källor"):
            for index, (text, source, score) in enumerate(similar_texts):
                st.write(f"Resultat {index + 1}:")
                st.write("Källa:", source)
                st.write("Träffsäkerhet:", score)
                st.write(f"Text från dokument:\n{text}")
                st.write("---")

    instructions_prompt = f"""
    Givet denna fråga: {user_input} och kontexten från en databas: {search_results}, sammanställ relevant fakta på ett lättläst sätt, samt ge ett utkast på hur ett svar skulle kunna se ut. Ditt svar riktas till en anställd på kommunen och skall utgöra ett stöd för den antsällde att återkoppla direkt till den som ställer frågan. Innehåller {user_input} både en fråga och synpunkt eller klagomål, addreserar du båda utifrån din fakta. Om du har fått rätt kontext i form av fakta för att ge ett korrekt svar så skriver du det, om inte så skriver du att kommunen har tagit emot synpunkten och diariefört den men att det inte är säkert att det finns resurser att prioritera just denna fråga. Inkludera källa för ditt svar. Svara vänligt men kortfattat.
    """

    answer = get_chat_response_streaming(user_input, instructions_prompt)

if 'response_completed' in st.session_state and st.session_state['response_completed']:

    with st.form(key='user_feedback_form', clear_on_submit=True):
        stars = st_star_rating("Hur nöjd är du med svaret", maxValue=5, defaultValue=3, key="rating")
        user_feedback = st.text_area("Vad var bra/mindre bra?")
        feedback_submit_button = st.form_submit_button("Skicka")
        print(user_input)

    if feedback_submit_button and user_input:
        print(st.session_state['record_id'])
        if 'record_id' in st.session_state and st.session_state['record_id']:
            update_data = {"user_rating": stars, "user_feedback": user_feedback}
            update_url = f"{directus_api_url}/{st.session_state['record_id']}"
            headers = {"Content-Type": "application/json"}
            params = {"access_token": st.secrets["directus_token"]}
            print(update_url)
            print(user_input)

            update_response = requests.patch(update_url, json=update_data, headers=headers, params=params)

            if update_response.status_code == 200:
                st.success("Tack för din feedback!")
                st.session_state['response_completed'] = False
                st.rerun()
            else:
                st.error("Något gick fel. Tack för din feedback!")

        


