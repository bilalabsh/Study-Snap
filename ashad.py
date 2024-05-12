import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv 
import string
import InstructorEmbedding
from InstructorEmbedding import INSTRUCTOR
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from htmlTemplates import css, user_template, bot_template
# from langchain.chat_models import ChatOpenAI

load_dotenv()

embeddings = OpenAIEmbeddings()
memory= ConversationBufferMemory(memory_key='chat_history',return_messages=True)
llm=ChatOpenAI()


def filter_text(text): #filter out unreadable tezt
    printable = set(string.printable)
    filtered_text = ''.join(filter(lambda x: x in printable, text))
    return filtered_text

def get_pdf_text(pdf_docs):
    text = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) 
        for page in pdf_reader.pages:
            text.extend(page.extract_text())
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.create_documents(text)
    return chunks

def helper(pdf_docs):
    text = get_pdf_text(pdf_docs)
    filtered_text = filter_text(text)
    text_chunks = get_chunks(filtered_text)
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vectorstore


def get_similar_docs(vector_store, user_question):
    
    # Assuming asimilarity_search returns a coroutine object
    similar_docs = vector_store.similarity_search(user_question, 3)[0].content
    
    # # Extract relevant terms from similar documents
    # relevant_terms = []
    # for doc in similar_docs:
    #     # Assuming doc['terms'] contains relevant terms for the document
    #     relevant_terms.extend(doc['terms'])
    
    # Create vector store from relevant terms
    # vectorstore = vector_store.(documents=, embedding=embeddings)
    
    return similar_docs

def get_conversation_chain(vectorstore):
    prompt_template = """You're a helpful and gentle assistant that answer questions based on documents chunks. \
    Answer the question based on the chat history (if any) and context below. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer.\
    If you think the information you are providing can be misleading and false, just accept that it is misleading. \
    Use three sentences maximum. Keep the answer as concise as possible. \
    CONTEXT: {context}
    Question: {question}
    Answer:"""

    CONV_QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CONV_QA_CHAIN_PROMPT},
        get_chat_history=lambda h : h
    )
    return conversation_chain

def conv(vectorstore, query):
    conv_qa = get_conversation_chain(vectorstore)
    response = conv_qa({"question": query})
    return response['answer']

def handle_userinput(user_question):
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if st.session_state.vectorstore is not None:
        response = conv(st.session_state.vectorstore, user_question)
        st.session_state.conversation.append(response)
        st.write(response)
    else:
        st.write("Please upload PDF documents first.")

    # Print conversation history
    for idx, msg in enumerate(st.session_state.conversation):
        st.write(f"Bot: {msg}")
    
    
    # st.write(bot_template.replace("{{MSG}}",response),unsafe_allow_html=True)
    # return response

# def handle_userinput(user_question):
#     response=conversation.get_response(user_question)
#     print(response)
#     return response


def main():
    
    st.set_page_config(page_title="Study Snap", page_icon="📚")
    st.write(css,unsafe_allow_html=True) 
    
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    
    
    st.header("Study Snap 📝")
    user_question=st.text_input("Enter a question about your documents: ")
    
    st.write(user_template.replace("{{MSG}}","Hello AI"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello User"),unsafe_allow_html=True)
    
    
    if user_question:
        if st.session_state.conversation is not None:  # Checking if conversation chain is initialized
            handle_userinput(user_question)
    
    
        
    with st.sidebar:
        st.subheader("Upload your documents")
        pdf_docs = st.file_uploader("Upload PDFs here and Click Enter", accept_multiple_files=True)
        
        if st.button("Enter"):
            with st.spinner("Processing your request"):
                # text = get_pdf_text(pdf_docs)
                # filtered_text = filter_text(text)
                # text_chunks = get_chunks(filtered_text)
                #create vector store
                # st.write(text_chunks)
                vectorstore=helper(pdf_docs)
                similar_docs= get_similar_docs(vectorstore, user_question)
                response = conv(vectorstore, user_question)
                st.session_state.conversation.append(response)
    
    # text = "My name is belol"

    # embeddings = HuggingFaceInstructEmbeddings()


    # embedded_text = embeddings.embed_query([text])

    # print("Embeddings:")
    # print(embedded_text)
   
    #get text chunks    
    
    
    print("vectorstore created")

    #allows to generate new msgs in convo
    
    
    
   

    #print(filtered_text)
    ##text_chunks = get_chunks(filtered_text)
    # print(text_chunks)
    # user_question=input("Enter your question: ")
    # if user_question:
    #     handle_userinput(user_question)
if __name__ == "__main__":
    main()
    
    
    
# def conversational_chat(self, k=5, chain_type="stuff"):
#         # Learn More: https://python.langchain.com/docs/use_cases/question_answering/how_to/chat_vector_db
#         """
#         Different chain types support different methods for combining retrieved documents:
#          - "map_reduce": Uses a map-reduce approach to combine documents. \
#             Each document is passed to the model separately and the answers are then combined.
#          - "stuff": Stuffs all the documents together into a single long document and passes that to the model.
#          - "qa_with_sources": Uses a Question Answering with Sources chain, \
#             which returns the answer along with the relevant source documents.
#         """

#         retriever = self.loader.db.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": k},
#         )
        # prompt_template = """You're a helpful and gentle assistant that answer questions based on documents chunks. \
        #     Answer the question based on the chat history (if any) and context below. \
        #     If you don't know the answer, just say that you don't know, don't try to make up an answer.\
        #     If you think the information you are providing can be misleading and false, just accept that it is misleading. \
        #     Use three sentences maximum. Keep the answer as concise as possible. \
        #     CONTEXT: {context}
        #     Question: {question}
        #     Answer:"""

        # CONV_QA_CHAIN_PROMPT = PromptTemplate(
        #     input_variables=["context", "question"],
        #     template=prompt_template,
        # )

#         conv_qa = ConversationalRetrievalChain.from_llm(
#             llm=ChatOpenAI(temperature=0.0),
#             chain_type=chain_type,
#             retriever=retriever,
#             combine_docs_chain_kwargs={"prompt": CONV_QA_CHAIN_PROMPT},
#             memory=self.memory,
#             get_chat_history=lambda h : h
#         )
#         return conv_qa

#     @timer
#     def conv_chat_response(self, query, k=5, chain_type="stuff"):
#         conv_qa = self.conversational_chat(k=k, chain_type=chain_type)
#         with get_openai_callback() as cb:
#             response = conv_qa({"question": query})
#             self.logger_1.info(cb)
#         self.questions_asked += 1
#         response = response['answer'] 
#         return response