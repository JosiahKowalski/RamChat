import os

import streamlit as st
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatGooglePalm, ChatOpenAI
from langchain_community.llms import Ollama, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from LLM import LLM
# from langchain_google_vertexai import VertexAI

import langchain
import GenerateVectorDB

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['GOOGLE_API_KEY'] = st.secrets['PALM_API_KEY']

if 'VectorDB' not in st.session_state:
    st.session_state.VectorDB = GenerateVectorDB.get_vector_db()
VectorStore = st.session_state.VectorDB

if 'Memory' not in st.session_state:
    st.session_state.Memory = ConversationBufferMemory(
        memory_key="history",
        input_key="question"
    )

CHAIN_CONFIG = [
    LLM(GoogleGenerativeAI, 'text-bison-001',
        'This is a general model best for straight forward questions, it will '
        'give a very short response to the question. Do not use this model.',
        {'model': 'models/text-bison-001'}),
    LLM(ChatOpenAI, 'gpt-3.5-turbo',
        'This is a general model best for straight forward questions, it will '
        'give a very short response to the question. Do not use this model.'),
    LLM(ChatGooglePalm, 'gemini-1.0-pro-001',
        'This is a chat model which will give very long responses to the query. '
        'This is best for questions which would need long explanations or supporting points. '
        'Use this model for all questions asking for lists or details.'),
    LLM(Ollama, 'mistral-7b',
        'This model is great for general purposes, '
        'it will give a medium length response to the question.',
        {'model': 'mistral'}),
    LLM(Ollama, 'codellama-7b',
        'This model is should only be used for coding related questions.',
        {'model': 'codellama'}),
    LLM(Ollama, 'gemma-7b', 'This is a Google model.', {'model': 'gemma'})
]


def initialize_llm(llm: any) -> object:
    """
    Initializes the LLM (Language Model) if it is not in the session.
    """
    if llm not in st.session_state:
        st.session_state[llm] = llm.get_retrieval_qa(vector_store=VectorStore, memory=st.session_state.Memory)
    return st.session_state[llm]


def select_best_model(user_input, exclude_model=None):
    llm = GoogleGenerativeAI(model="models/text-bison-001")  # GooglePalm is good and normally gives one-word responses
    prompt = (
        f"Given the user question: '{user_input}', evaluate which of the following models is most suitable: "
        f"Strictly respond in 1 word only."
    )
    for model in CHAIN_CONFIG:
        if model != exclude_model:  # Exclude the specified model from consideration
            prompt += f"\n- {model}: {model.description}"

    print(prompt)
    # Send prompt to model with no chain
    llm_response = llm.invoke(input=prompt)

    # Parse the response to find the best model
    # This part depends on how your LLM formats its response. You might need to adjust the parsing logic.
    best_model = parse_llm_response(llm_response)

    return best_model


def parse_llm_response(response):
    # Convert response to lower case for case-insensitive matching
    response_lower = response.lower()

    # Initialize a dictionary to store the occurrence count of each model in the response
    model_occurrences = {model: response_lower.count(model.model_name.lower()) for model in CHAIN_CONFIG}

    # Find the model with the highest occurrence count
    best_model = max(model_occurrences, key=model_occurrences.get)

    print("best model:", best_model)

    # If no model is mentioned or there is a tie, you might need additional logic to handle these cases
    if model_occurrences[best_model] == 0:
        print("No model found, using ChatGooglePalm")
        return CHAIN_CONFIG[2]  # Or some default model

    return best_model


def generate_response(input_text, selected_chain):
    """
    Takes an input text and a language model chain to generate a response.

    It constructs a query from the input text, passes it on to the selected language model,
    and captures the model's response. The response typically includes the output
    and meta-information about the source documents that the model used to generate the response.

    :param input_text: str, The input text prompt that is used to generate a response.
    :param selected_chain: str, The selected language model from the chain configuration.

    :return: tuple(str, str),
             The first element represents the generated response from the language model,
             and the second element is the URL of the source.
    """
    response = selected_chain({'query': input_text})
    source_docs = response['source_documents']
    if len(source_docs) > 0:
        return response['result'], source_docs[0].metadata['source']
    return response['result'], None


def write_response(query):
    selected_model = select_best_model(text)
    chain = initialize_llm(selected_model)
    try:
        response, source = generate_response(query, chain)
    except Exception as e:
        print(f"Error: {e}")
        print("Switching to another model...")

        # Select another model as a fallback
        fallback_model = select_best_model(query, exclude_model=selected_model)
        print("Selected fallback model:", fallback_model)

        # Re-initialize the fallback model
        chain = initialize_llm(fallback_model)
        response, source = generate_response(query, chain)
    st.sidebar.write(f"Model: {selected_model}")
    st.write(response)
    if source is not None:
        st.write(source)


st.title('🐏💬 RamChat')

with st.form('my_form'):
    placeholder = 'What is the snow policy?'
    text = st.text_area('Enter text:', placeholder=placeholder)
    submitted = st.form_submit_button('Submit')
    if submitted:
        text = text or placeholder
        write_response(text)
