from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


class LLM:
    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    def __init__(self, model, caption, description, opts=None):
        if opts is None:
            opts = {}
        self.opts = opts
        self.model_name = model.__name__ if model.__name__ != 'Ollama' else opts['model']
        self.model = model
        self.caption = caption
        self.description = description

    def __str__(self):
        return self.model_name

    def get_retrieval_qa(self, vector_store, memory):
        chain = RetrievalQA.from_chain_type(llm=self.model(**self.opts),
                                            chain_type='stuff',
                                            retriever=vector_store.as_retriever(),
                                            return_source_documents=True,
                                            chain_type_kwargs={
                                                "prompt": LLM.prompt,
                                                "memory": memory
                                            })
        return chain
