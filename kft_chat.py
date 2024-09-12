import requests
import streamlit as st
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from streamlit_star_rating import st_star_rating
import datetime
import html

current_date = datetime.date.today()

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(page_title="Kommunens Chatbot", page_icon=":robot_face:", layout="wide")

openai_api_key = st.secrets['OPENAI_API_KEY']
qdrant_api_key = st.secrets['qdrant_api_key']
qdrant_url = "https://qdrant.utvecklingfalkenberg.se"
collection_name2 = "FalkenbergsKommunsHemsida_1000char_chunks"
collection_name3 = "mediawiki"
directus_api_url = "https://nav.utvecklingfalkenberg.se/items/kft_bot"
directus_params = {"access_token": st.secrets['directus_token']}

openai_client = OpenAI(api_key=openai_api_key)
qdrant_client = QdrantClient(url=qdrant_url, port=443, https=True, api_key=qdrant_api_key)

def generate_embeddings(text):
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-large")
    return response.data[0].embedding

def search_collection(qdrant_client, collection_name, user_query_embedding):
    response = qdrant_client.search(collection_name=collection_name, query_vector=user_query_embedding, limit=10, with_payload=True)
    return response

def get_chat_response_streaming(user_message, extra_knowledge, instructions_prompt, model="gpt-4o", client=None):
    if client is None:
        client = OpenAI(api_key=openai_api_key)
    full_response = ""
    message_placeholder = st.empty()
    completion = client.chat.completions.create(model=model, messages=[{"role": "system", "content": instructions_prompt}], stream=True)
    for chunk in completion:
        st.session_state['response_completed'] = False
        if chunk.choices[0].finish_reason == "stop":
            st.session_state['response_completed'] = True
            message_placeholder.markdown(full_response)
            data = {"prompt": user_message, "instruction_prompt": extra_knowledge, "response": full_response}
            directus_response = requests.post(directus_api_url, json=data, params=directus_params)
            if directus_response.status_code == 200:
                response_data = directus_response.json()
                st.session_state['record_id'] = response_data['data']['id']
                print(st.session_state['record_id'])
            else:
                st.error("Något gick fel. Försök igen senare.")
            return full_response
        full_response += chunk.choices[0].delta.content
        message_placeholder.markdown(full_response + "▌")

def clear_form():
    st.session_state['user_input'] = ""
    st.session_state['extra_knowledge'] = ""
    if 'use_hemsidan' in st.session_state:
        del st.session_state['use_hemsidan']
    if 'use_mediawiki' in st.session_state:
        del st.session_state['use_mediawiki']

st.title("Demo KFT - utkastsgenererare")

if st.button("Rensa formulär"):
    clear_form()

with st.form(key='user_query_form', clear_on_submit=False):
    user_input = st.text_area("Klistra in fråga/klagomål från invånare här:", key="user_input", height=100)
    st.caption("Svaren genereras av en AI-bot, som kan begå misstag. Frågor och svar lagras i utvecklingssyfte. Skriv inte personuppgifter i fältet.")
    extra_knowledge = st.text_area("Klistra in extra kontext/kunskap/fakta/instruktioner här:", key="extra_knowledge", height=100, placeholder="Hänvisa till Falkenbergs kommuns bestämmelser för stöd till föreningsvlivet och bla bla bla")

    st.write("Välj vilka samlingar som ska användas för sökning:")
    use_hemsidan = st.checkbox("Falkenbergs kommuns hemsida", value=True, key="use_hemsidan")
    use_mediawiki = st.checkbox("MediaWiki - KFT: Intern dokumentation", value=True, key="use_mediawiki")

    input_to_embed = user_input + extra_knowledge
    submit_button = st.form_submit_button("Genera utkast till svar 🪄")

if submit_button and user_input:
    user_embedding = generate_embeddings(input_to_embed)
    combined_results = []
    used_collections = []

    if use_hemsidan:
        search_results2 = search_collection(qdrant_client, collection_name2, user_embedding)
        for result in search_results2:
            source_info = html.escape(f"{result.payload['title']} (URL: {result.payload['url']})")
            combined_results.append({
                'text': html.escape(result.payload['chunk']),
                'source': source_info,
                'score': result.score,
                'category': 'hemsidan'
            })
        used_collections.append("Falkenbergs kommuns hemsida")

    if use_mediawiki:
        search_results3 = search_collection(qdrant_client, collection_name3, user_embedding)
        for result in search_results3:
            combined_results.append({
                'text': html.escape(result.payload['chunk']),
                'source': html.escape(result.payload['title']),
                'score': result.score,
                'category': 'interndokumentation'
            })
        used_collections.append("Intern dokumentation")

    ranked_results = sorted(combined_results, key=lambda x: x['score'], reverse=True)
    
    # Store the ranked results in the session state
    st.session_state['ranked_results'] = ranked_results
    st.session_state['used_collections'] = used_collections

    context_from_db = ", ".join([
        f"{result['text']} (Category: {result['category']}, Source: {result['source']})"
        for result in ranked_results])

    instructions_prompt = f"""
Givet denna invånar-fråga: '{user_input}', samt om det finns ytterligare information från kommunanställd 'extra-instruktioner': {extra_knowledge}, samt kontexten från en databas: {context_from_db}, sammanställ relevant fakta på ett lättläst sätt, samt ge ett utkast på hur ett svar skulle kunna se ut. Ditt svar riktas till en anställd på kommunen och ska utgöra ett stöd för den anställde att återkoppla direkt till den som ställer frågan. Innehåller {user_input} både en fråga och en synpunkt eller klagomål, adressera du båda utifrån din fakta. Om du har rätt kontext i form av fakta för att ge ett korrekt svar så skriver du det, om inte så skriver du att kommunen har tagit emot synpunkten och diariefört den men att det inte är säkert att det finns resurser att prioritera just denna fråga. Inkludera källa för ditt svar.

Ibland kan du få in information som säger 'i år bla bla', men texten är äldre då den kan vara publicerad som en nyhet. Känn till dagens datum: {current_date}, så kan du själv avgöra om texten är helt aktuell, eller åtminstone referera till eventuellt datum i ditt svar. Till exempel kan man istället för att säga 'i år', så hade man kunnat skriva det aktuella datumet.

Svara vänligt men kortfattat.
Ditt svar börjar med: 'Hej Namn,' avslutas med: 'Med vänliga hälsningar, [Namn], [Avdelning på kommunen]'
Oavsett du har rätt fakta eller inte till invånaren ska du svara koncist, to the point men professionellt artigt. Du måste hänvisa till källan du baserar ditt svar. Finns det kontaktpersoner och kontaktuppgifter, inkludera gärna dessa. Skriv gärna hur många av de underlag som du fick till dig som du använt för ditt svar.

Observera att följande samlingar användes för sökningen: {', '.join(used_collections)}.
"""

    st.session_state['original_response'] = get_chat_response_streaming(user_input, extra_knowledge, instructions_prompt)

# Display the sources in the expander with two columns
if 'ranked_results' in st.session_state and st.session_state['ranked_results']:
    with st.expander("Se relevanta källor"):
        col1, col2 = st.columns(2)
        for index, result in enumerate(st.session_state['ranked_results']):
            with col1 if index % 2 == 0 else col2:
                st.write(f"Resultat {index + 1}:")
                st.write("Källa:", result['source'])
                st.write("Träffsäkerhet:", result['score'])
                st.write(f"Text från dokument:\n{result['text']}")
                st.write("---")

# Modify the section handling the "Förbättra svaret" button
if 'original_response' in st.session_state and st.session_state['original_response']:
    st.markdown("### Ytterligare instruktioner")
    additional_instructions = st.text_area("Ange ytterligare instruktioner för att förbättra svaret:", key="additional_instructions")
    
    if st.button("Förbättra svaret"):
        new_instructions_prompt = f"""
        Här är det ursprungliga svaret:

        {st.session_state['original_response']}

        Användaren har gett följande ytterligare instruktioner: {additional_instructions}

        Vänligen modifiera det ursprungliga svaret baserat på dessa instruktioner. Håll det ursprungliga formatet med 'Hej Namn,' i början och 'Med vänliga hälsningar, [Namn], [Avdelning på kommunen]' i slutet.
        """
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Ursprungligt svar:")
            st.write(st.session_state['original_response'])
        with col2:
            st.markdown("### Förbättrat svar")
            st.session_state['improved_response'] = get_chat_response_streaming("", "", new_instructions_prompt)
        st.session_state['response_completed'] = True
        st.session_state['final_response'] = st.session_state['improved_response']

if 'response_completed' in st.session_state and st.session_state['response_completed']:
    with st.form(key='user_feedback_form', clear_on_submit=True):
        stars = st_star_rating("Hur nöjd är du med svaret", maxValue=5, defaultValue=3, key="rating")
        user_feedback = st.text_area("Vad var bra/mindre bra?")
        feedback_submit_button = st.form_submit_button("Skicka")

    if feedback_submit_button:
        if 'record_id' in st.session_state and st.session_state['record_id']:
            update_data = {
                "user_rating": stars, 
                "user_feedback": user_feedback,
                "response": st.session_state.get('final_response', st.session_state['original_response'])
            }
            update_url = f"{directus_api_url}/{st.session_state['record_id']}"
            headers = {"Content-Type": "application/json"}
            params = {"access_token": st.secrets["directus_token"]}

            update_response = requests.patch(update_url, json=update_data, headers=headers, params=params)

            if update_response.status_code == 200:
                st.success("Tack för din feedback!")
                st.session_state['response_completed'] = False
                st.rerun()
            else:
                st.error("Något gick fel. Tack för din feedback!")