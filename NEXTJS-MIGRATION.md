# Starter-prompt: porta KFT-boten till Next.js

> **Till Claude:** Läs hela den här filen innan du gör något. Den beskriver vad
> som finns i dag och vad som ska byggas. Uppgifterna är kontrollerade mot
> koden 2026-10-28. Läs de källfiler som nämns innan du drar slutsatser och
> fråga Tomas när något är oklart. Börja med en plan och bygg inget förrän han
> har godkänt den.

## Bakgrund

KFT-boten ("KFT utkastgenereraren") hjälper kommunanställda i Falkenberg att
skriva svar till invånare. Den anställda klistrar in en medborgarfråga. Boten
söker i kommunens hemsida och interna wiki och ger tillbaka fakta och ett
brevutkast med källor.

I dag körs den som en Streamlit-app på Streamlit Community Cloud
(https://kft-bot.streamlit.app). Appen deployas från `main` i det här repot.
Målet är att göra om den till en riktig Next.js-app.

Systemet består av två repon:

| Repo | Vad | Fil som räknas |
|---|---|---|
| `cryptonicsurfer/kft-bot` (det här) | Chatten och brevutkasten | **`chat_with_letter_tools.py`** |
| `frejandreassen/kft-update` | Bygger om wiki-indexet med en knapptryckning | `uppdatera_fran_mediawiki.py` |

**Övriga `.py`-filer i det här repot är experiment eller äldre versioner.**
Portera dem inte. `kft_chat.py` visar bara en länk till den nya adressen.

## Så fungerar boten i dag (`chat_with_letter_tools.py`)

- **LLM:** OpenAI `gpt-4o`, `temperature=0.2`, streaming och tool calling
  (`tool_choice="auto"`). **Den nya appen ska inte använda `gpt-4o`.** Den
  ska troligen använda `gpt-6.1-sol`, eller `gpt-6.1-terra`/`gpt-6.1-luna`
  (beslutet är inte slutligt, så fråga Tomas). Lägg modell-id:t i en env-variabel
  (`OPENAI_MODEL`) så att det går att byta utan kodändring. Kontrollera i
  OpenAI:s dokumentation vilka parametrar modellen stöder (t.ex. `temperature`
  och reasoning-inställningar) och om Responses API är rätt väg i stället för
  Chat Completions.
- **Systemprompt:** konstanten `SYSTEM_MESSAGE` i filen. Flytta över den
  ordagrant som utgångspunkt. Den styr bland annat:
  - brevet börjar med "Hej Namn," och slutar med "Med vänliga hälsningar,
    [Namn], [Avdelning på kommunen]"
  - brevet ligger inom `<letter>…</letter>` med källänkar efter den avslutande
    taggen
  - boten hänvisar aldrig till en enskild person, utan till kontaktcenter
    (0346-88 60 00, kontaktcenter@falkenberg.se)
  - när fakta saknas skriver boten att synpunkten är diarieförd
- **Verktyg 1, `search_qdrant(user_input, limit=3)`:**
  - skapar en embedding med `text-embedding-3-large` (3072 dimensioner)
  - söker med `query_points` i två Qdrant-collections, `FalkenbergsKommunsHemsida`
    och `mediawiki`, med `limit` träffar från varje
  - returnerar `score` och `payload` till modellen
- **Verktyg 2, `submit_feedback(user_rating 1–5, user_feedback)`:**
  - skickar en POST till Directus `https://nav.utvecklingfalkenberg.se/items/kft_bot`
    med `access_token` som query-param
  - body: `{prompt: <chatthistorik utan function-meddelanden>, user_rating, user_feedback}`
- **UI:** två kolumner. Till vänster finns chatten, där brevet visas i en
  utfällbar "Brev"-ruta. Till höger visas det senaste brevet, och det fylls på
  medan det streamas. Högst upp står varningen: "skriv inte in personuppgifter
  eller känslig information".
- **Teknisk skuld som inte ska följa med:**
  - koden använder det gamla `role: "function"`-formatet (använd `tool`-meddelanden)
  - `safe_json_loads` lappar ihop tool-argument som har slagits ihop med regex
  - toolloopen stöder bara ett verktygsanrop per tur
  - allt state ligger i `st.session_state`

## Så fungerar wiki-uppdateringen i dag (`kft-update`)

- Loggar in på MediaWiki `https://wiki01.utvecklingfalkenberg.se/api.php`
  som bot-användaren `Api01@api`.
- Hämtar alla sidor (`list=allpages`, max 1000) och deras senaste revision.
- **Raderar hela collectionen `mediawiki`** och skapar den på nytt
  (3072 dimensioner, cosine).
- Delar texten i chunkar på 4500 tecken med 500 tecken överlapp. Varje
  chunk-text börjar med `Title: … Content: …`.
- Varje punkt har en payload med `title`, `chunk`, `chunk_info`, `link_to_article`
  och `last_revised`. Punktens id är ett UUID från md5 av chunk-texten.
- Embeddings skapas i batchar om 200.
- **Problem som ska lösas i Next.js-versionen:**
  - vem som helst med länken kan trycka på knappen
  - under ombyggnaden är collectionen tom
  - om något går fel halvvägs blir den tom eller ofullständig
  - mer än 1000 sidor hämtas inte (paginering saknas)
  - Bygg om i en ny collection och byt sedan med en Qdrant-alias, eller gör
    en inkrementell uppdatering via `last_revised`. Kör den som en skyddad
    API-route eller ett cron-jobb, inte som en publik knapp.

`FalkenbergsKommunsHemsida` byggs **inte** av något av de här två repona.
Troligen kommer den från `frejandreassen/falkenbergs-kommuns-hemsida`, men det
är okontrollerat. Läs inte och skriv inte till den collectionen förrän Tomas
vet hur den fylls på. Jämför först payload-formatet med en riktig punkt.

## Hemligheter

Det finns inga nycklar i repona. De ligger i Streamlit-panelens secrets och
lämnas över av Paul eller Frej. Namnen i dag:

- kft-bot: `OPENAI_API_KEY`, `qdrant_url`, `qdrant_api_key`, `directus_token`
- kft-update: `OPENAI_API_KEY`, `QDRANT_API_KEY`, `MEDIAWIKI_API_KEY`

I Next.js ska alla nycklarna stanna på serversidan. Använd aldrig ett
`NEXT_PUBLIC_`-prefix för dem. Anropa OpenAI, Qdrant och Directus bara från
route handlers eller server actions. Lägg till `.env*` i `.gitignore` och, om
appen ska köras i Docker, även i `.dockerignore`.

## Krav på den nya appen

1. **Funktionsparitet först:** samma systemprompt, samma två verktyg, samma
   collections och samma embedding-modell. Chattmodellen byts (se ovan). Frågevektorerna **måste** skapas med
   `text-embedding-3-large`, annars matchar de inte indexet.
2. **Streaming** till klienten, med brevet i en egen panel bredvid chatten som
   i dag. Gör det lätt att kopiera brevet.
3. **Feedback** till samma Directus-collection `kft_bot` med samma fält, så att
   den historik som redan finns går att jämföra med nya svar.
4. **Inloggning:** boten är till för anställda och ska inte vara öppen på
   internet. Fråga Tomas vilken lösning som gäller, till exempel kommunens
   intranät-SSO.
5. **Wiki-uppdateringen** som en skyddad server-funktion (se problemen ovan).
6. **Lås versionerna** av beroenden som hanterar nycklar (`openai`,
   `@qdrant/js-client-rest` m.fl.) till exakta versioner utan `^`, och använd
   `npm ci` när lockfilen väl finns.

## Förslag på första steg

1. Läs `chat_with_letter_tools.py` och `uppdatera_fran_mediawiki.py` i
   `frejandreassen/kft-update` från början till slut.
2. Hämta en punkt ur var och en av collectionerna och kontrollera hur
   payloaden ser ut. Gör det bara med läsanrop.
3. Föreslå en struktur för appen (routes, var verktygen ligger, hur streaming
   och tool-loop fungerar, auth) och vänta på att Tomas säger ok.
4. Bygg chatten med paritet. Jämför sedan några riktiga medborgarfrågor sida
   vid sida med Streamlit-versionen innan något annat läggs till.

Stäng inte av Streamlit-appen förrän den nya appen är i drift och har
godkänts. Den används i dag.
