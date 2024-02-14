import gc
import os

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatGooglePalm, ChatOpenAI
from langchain_community.llms import Ollama, OpenAI
from langchain_google_genai import GoogleGenerativeAI
# from langchain_google_vertexai import VertexAI

import langchain
import GenerateVectorDB

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['GOOGLE_API_KEY'] = st.secrets['PALM_API_KEY']

if 'VectorDB' not in st.session_state:
    st.session_state.VectorDB = GenerateVectorDB.get_vector_db()
VectorStore = st.session_state.VectorDB

CHAIN_CONFIG = {
    # for some reason this is giving terrible responses
    'OpenAI': {'llm_class': OpenAI, 'llm_opts': {'temperature': 0.1, 'model': 'davinci-002'}, 'caption': 'davinci-002',
               'llm_description': 'This is a general model best for questions. Do not use this model.'},
    'GoogleGenerativeAI': {'llm_class': GoogleGenerativeAI, 'caption': 'text-bison-001',
                           'llm_opts': {'model': 'models/text-bison-001'},
                           'llm_description': 'This is a general model best for straight forward questions, it will '
                                              'give a very short response to the question. Do not use this model.'},
    'ChatOpenAI': {'llm_class': ChatOpenAI, 'caption': 'gpt-3.5-turbo',
                   'llm_description': 'This is a chat model best for conversational questions or requests. Do not use '
                                      'this model'},
    'ChatGooglePalm': {'llm_class': ChatGooglePalm, 'caption': 'chat-bison-001',
                       'llm_description': 'Always use this model. This is a chat model which will give very long '
                                          'responses to the query.'
                                          'This is best for questions which would need long explanations or '
                                          'supporting points. Use this model for all questions asking for lists or '
                                          'details.'},
    'Mistral': {'llm_class': Ollama,
                'llm_opts': {'model': 'mistral'}, 'caption': 'mistral-7b',
                'llm_description': 'Always use this model. This model is great for general purposes, it will give a '
                                   'medium length response to the question.'},
    'CodeLlama': {'llm_class': Ollama,
                  'llm_opts': {'model': 'codellama'}, 'caption': 'llama-7b',
                  'llm_description': 'This model is should only be used for coding related questions.'}
}


def cleanup_llm(llm_name):
    """
    This method cleans up the specified Large Language Model (LLM) by removing it from the session state and
    performing garbage collection to explicitly free up unreferenced memory.

    :param llm_name: The name of the Large Language Model (LLM) to be cleaned up.
    :return: None
    """
    if llm_name in st.session_state:
        del st.session_state[llm_name]
        gc.collect()  # Explicitly free up unreferenced memory


def initialize_llm(llm_name: str, llm_class: any, llm_opts: dict) -> object:
    """
    Initializes the LLM (Language Model) if it is not in the session.

    :param llm_name: Name of the LLM.
    :type llm_name: str
    :param llm_class: Class of the LLM.
    :type llm_class: class
    :param llm_opts: Options for the LLM initialization.
    :type llm_opts: dict
    :return: The initialized LLM.
    :rtype: object
    """

    # initialize the llm if it is not in the session
    if llm_name not in st.session_state:
        # clear memory if localLLM, localLLMs contain a model_path
        if 'model_path' in llm_opts:
            for chain_key in CHAIN_CONFIG.keys():
                # avoid self checking
                if chain_key != llm_name and 'model_path' in CHAIN_CONFIG[chain_key].get('llm_opts', {}):
                    cleanup_llm(chain_key)
        llm = llm_class(**llm_opts)
        st.session_state[llm_name] = RetrievalQA.from_llm(llm=llm, retriever=VectorStore.as_retriever(),
                                                          return_source_documents=True)
        # noinspection PyTypeChecker
        st.session_state.selected_radio = list(CHAIN_CONFIG.keys()).index(llm_name)
        st.rerun()  # this causes it to rerun, this was here due to being able to select models
    return st.session_state[llm_name]


def init_chain_from_config(selected_chain):
    """
    Initialize a chain from the given configuration.

    :param selected_chain: The selected chain from the configuration.
    :return: The initialized chain.
    """
    config = CHAIN_CONFIG[selected_chain]
    init_chain = initialize_llm(selected_chain,
                                config['llm_class'],
                                config.get('llm_opts', {}))  # Optional key. Returns empty dict if not found.
    return init_chain


def select_best_model(user_input):
    llm = GoogleGenerativeAI(model="models/text-bison-001")  # GooglePalm is good and normally gives one word responses
    prompt = (f"Given the user question: '{user_input}', evaluate which of the following models is most suitable: "
              f"Strictly respond in 1 word only.")
    for model, description in CHAIN_CONFIG.items():
        prompt += f"\n- {model}: {description['llm_description']}"

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
    # TODO: THIS DOES NOT WORK WELL, It finds GooglePalm when the response is ChatGooglePalm, need to fix
    model_occurrences = {model: response_lower.count(model.lower()) for model in CHAIN_CONFIG}

    # Find the model with the highest occurrence count
    best_model = max(model_occurrences, key=model_occurrences.get)

    print("best model:", best_model)

    # If no model is mentioned or there is a tie, you might need additional logic to handle these cases
    if model_occurrences[best_model] == 0:
        print("No model found, using OpenAI")
        return "OpenAI"  # Or some default model

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
        # seems like this doesn't return the most accurate source doc since it is getting the first one
        print("Source documents", source_docs)
        return response['result'], source_docs[0].metadata['source']
    return response['result'], None


st.title('🦜🔗 Shepherd University Demo')
captions = []
for key in CHAIN_CONFIG.keys():
    caption = CHAIN_CONFIG[key]['caption']
    if key in st.session_state:
        caption += ':green[ - Ready]'
    captions.append(caption)
with st.sidebar:
    selected_llm = st.radio('Choose a LLM to use: ',
                            list(CHAIN_CONFIG.keys()),
                            captions=captions,
                            index=0 if 'selected_radio' not in st.session_state else st.session_state.selected_radio)

chain = init_chain_from_config(selected_llm)

with st.form('my_form'):
    placeholder = 'What is rule 1000 in the code of conduct?'
    text = st.text_area('Enter text:', placeholder=placeholder)
    submitted = st.form_submit_button('Submit')
    if submitted:
        text = text or placeholder
        # selected_model = select_best_model(text)
        # print(selected_model)
        # chain = init_chain_from_config(selected_model)
        answer, source = generate_response(text, chain)
        # st.write(selected_model + answer)
        st.write(answer)
        if source is not None:
            st.write(source)
