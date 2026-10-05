# Study Snap

Chat with your study material. Upload PDFs and ask questions about them. Study Snap finds the relevant passages and answers from your documents, remembering the conversation so you can ask follow-ups.

## How it works

1. PDFs are loaded, cleaned and split into chunks
2. Chunks are embedded and stored in a Chroma vector database
3. Each question retrieves the most relevant chunks, and an OpenAI chat model answers from them (retrieval-augmented generation)
4. Conversation memory keeps follow-up questions in context

## Tech stack

Python · LangChain · Chroma · OpenAI (embeddings and chat) · Streamlit

## Run it

```bash
pip install -r requirements.txt
# add OPENAI_API_KEY to a .env file
streamlit run main.py
```
