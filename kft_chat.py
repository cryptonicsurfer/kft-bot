import requests
import streamlit as st
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from streamlit_star_rating import st_star_rating
import datetime

current_date = datetime.date.today()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

openai_api_key = st.secrets['OPENAI_API_KEY']
qdrant_api_key = st.secrets['qdrant_api_key']
qdrant_url = "https://qdrant.utvecklingfalkenberg.se"
# collection_name = "KFT_knowledge_base_OpenAI_Large_chunk1000"
collection_name2="FalkenbergsKommunsHemsida_1000char_chunks" #_1000char_chunks 
collection_name3="mediawiki"
directus_api_url = "https://nav.utvecklingfalkenberg.se/items/kft_bot"
directus_params={"access_token":st.secrets['directus_token']}

openai_client = OpenAI(api_key=openai_api_key)
qdrant_client = QdrantClient(url=qdrant_url, port=443, https=True, api_key=qdrant_api_key)

def generate_embeddings(text):
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-large")
    return response.data[0].embedding

def search_collection(qdrant_client, collection_name, user_query_embedding):
    response = qdrant_client.search(collection_name=collection_name, query_vector=user_query_embedding, limit=3 , with_payload=True) #score_threshold=0.4)
    return response

#gpt-4-turbo-preview
def get_chat_response_streaming(user_message, extra_knowledge, instructions_prompt, model="gpt-4o", client=None):
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
            data={"prompt":user_message, "instruction_prompt": extra_knowledge, "response": full_response}
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
# st.write(collection_name)
with st.form(key='user_query_form', clear_on_submit=True):
    user_input = st.text_area("Klistra in fråga/klagomål från invånare här:", key="user_input", height=100)
    st.caption("Svaren genereras av en AI-bot, som kan begå misstag. Frågor och svar lagras i utvecklingssyfte. Skriv inte personuppgifter i fältet.")
    extra_knowledge = st.text_area("Klistra in extra kontext/kunskap/fakta/instruktioner här:", key="extra_knowledge", height=100)
    input_to_embed = user_input + extra_knowledge
    submit_button = st.form_submit_button("Genera utkast till svar 🪄")


if submit_button and user_input:
    user_embedding = generate_embeddings(input_to_embed)
    # search_results = search_collection(qdrant_client, collection_name, user_embedding)
    search_results2 = search_collection(qdrant_client, collection_name2, user_embedding)
    search_results3=search_collection(qdrant_client,  collection_name3, user_embedding)

    
    combined_results = []

    # Handling search_results1
    # for result in search_results:
    #     combined_results.append({
    #         'text': result.payload['text'],  # Using text for content
    #         'source': result.payload['file_source'],  # Using file_source as source
    #         'score': result.score,
    #         'category': 'kft-filer'
    #     })
        
    # Handling search_results2
    for result in search_results2:
        source_info = f"{result.payload['title']} (URL: {result.payload['url']})"
        combined_results.append({
            'text': result.payload['chunk'],  # Using chunk for content
            'source': source_info,  # Combining title and url for source
            'score': result.score,
            'category': 'hemsidan'
        })

    # Handling search_results3
    for result in search_results3:
        combined_results.append({
            'text': result.payload['chunk'],  # Using chunk for content
            'source': result.payload['title'],  # Using title as source
            'score': result.score,
            'category': 'interndokumentation'
        })

    # Sorting combined results based on score in descending order
    ranked_results = sorted(combined_results, key=lambda x: x['score'], reverse=True)




    if ranked_results:
        with st.expander("Se relevanta källor"):
            for index, result in enumerate(ranked_results):
                st.write(f"Resultat {index + 1}:")
                st.write("Källa:", result['source'])
                st.write("Träffsäkerhet:", result['score'])
                st.write(f"Text från dokument:\n{result['text']}")
                st.write("---")

    # Uppdatera instruktionsprompt med dynamisk kontext från sorterade resultatsatser
    # context_from_db = ", ".join([f"{result['text']}" for result in ranked_results])
    # context_from_db = ", ".join([f"{result['text']} (Category: {result['category']})" for result in ranked_results])
    # Constructing context_from_db
    context_from_db = ", ".join([
        f"{result['text']} (Category: {result['category']}, Source: {result['source']})"
        for result in ranked_results])



    instructions_prompt = f"""
Givet denna invånar-fråga: '{user_input}', samt om det finns ytterligare information från kommunanställd 'extra-instruktioner': {extra_knowledge}, samt kontexten från en databas: {context_from_db}, sammanställ relevant fakta på ett lättläst sätt, samt ge ett utkast på hur ett svar skulle kunna se ut. Ditt svar riktas till en anställd på kommunen och ska utgöra ett stöd för den anställde att återkoppla direkt till den som ställer frågan. Innehåller {user_input} både en fråga och en synpunkt eller klagomål, adressera du båda utifrån din fakta. Om du har rätt kontext i form av fakta för att ge ett korrekt svar så skriver du det, om inte så skriver du att kommunen har tagit emot synpunkten och diariefört den men att det inte är säkert att det finns resurser att prioritera just denna fråga. Inkludera källa för ditt svar.

Ibland kan du få in information som säger 'i år bla bla', men texten är äldre då den kan vara publicerad som en nyhet. Känn till dagens datum: {current_date}, så kan du själv avgöra om texten är helt aktuell, eller åtminstone referera till eventuellt datum i ditt svar. Till exempel kan man istället för att säga 'i år', så hade man kunnat skriva det aktuella datumet.
    
Svara vänligt men kortfattat.
Ditt svar börjar med: 'Hej Namn,' avslutas med: 'Med vänliga hälsningar, [Namn], [Avdelning på kommunen]'
Oavsett du har rätt fakta eller inte till invånaren ska du svara koncist, to the point men professionellt artigt. Du måste hänvisa till källan du baserar ditt svar. Finns det kontaktpersoner och kontaktuppgifter, inkludera gärna dessa. Skriv gärna hur många av de underlag som du fick till dig som du använt för ditt svar. 

"""

    # Presume get_chat_response_streaming takes the updated instructions prompt
    answer = get_chat_response_streaming(user_input, extra_knowledge, instructions_prompt)
    # print(ranked_results)
    # print('_'*30)
    # print(context_from_db)


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

        


