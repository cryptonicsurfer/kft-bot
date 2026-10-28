# KFT Bot

Streamlit-app för att hjälpa kommunanställda skriva svar till invånare. Deployad via Streamlit Community Cloud (hämtar automatiskt från GitHub main).

**Ska portas till Next.js** — läs `NEXTJS-MIGRATION.md` innan du gör något annat.

## Tech Stack
- Streamlit
- OpenAI (GPT-4o i dag → gpt-6.1-sol/terra/luna i Next.js-versionen; text-embedding-3-large)
- Qdrant (vektorsök)
- Directus CMS (feedback)

## Känd teknisk skuld

### Qdrant API-migrering (qdrant-client v1.14+)
`search()` togs bort i qdrant-client v1.14 och ersattes med `query_points()`. Fixen gjordes i `chat_with_letter_tools.py` men följande filer använder fortfarande det gamla API:t och behöver samma fix:
- `streaming_kft_chat_artefacts.py:46`
- `kft_chat_artefacts.py:46`
- `column_test.py:31`

Fix: `search(collection_name, query_vector=..., limit, with_payload)` → `query_points(collection_name, query=..., limit, with_payload)` och `.points` på svaret.
