from langchain_openai import ChatOpenAI
from langchain_core import runnables, documents
from langchain_core.messages.ai import AIMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from typing import List
import itertools

# TODO: Bert-base-uncased Tokenizer not working, Transformers broken install
# import transformers.generation.utils
# import transformers

# tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
# text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
#     tokenizer, chunk_size=8000, chunk_overlap=100
# )

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)


# Return Valid Topics to search in Wikipedia
def res_to_topics(res: AIMessage) -> list:
    res_topics = []
    for topic in res.content.split(","):
        topic_str = topic.strip().replace("_", " ")
        if topic_str.lower()=="<unretrievableerror>":
            continue
        res_topics.append(topic_str)

    return res_topics


# Prompts the Model to return at most 5 topics relevant to the search query.
# Returns a Chain
def fetch_wiki_topics(llm: ChatOpenAI):
    template = \
    """
    Context: You are a Wikipedia researcher. You list at most 5 relevant topics, including basics, to a query asked. 
    Only concise searchable topic headings at most 4 words, nothing else. If you don't know the answer, just say "<unretrievableerror>".
    Strictly structure the output as comma seperated values.

    Query: {query}
    """

    wiki_topic_prompt = ChatPromptTemplate.from_template(template)
    
    return wiki_topic_prompt | llm | res_to_topics


# Gets Wikipedia Documents for a topic
# Splits the documents and returns a list of documents split according to chunk size
def fetch_doc_and_split(topic: str) -> List:
    wiki_docs = WikipediaLoader(query=topic, load_max_docs=2).load()
    return text_splitter.split_documents(wiki_docs)


# For all topics to search, get the document and do split parallelly
# Return a list of split Documents for all topics
def get_all_topics_docs(topics: list) -> list[documents.Document]:
    runnable_get_doc_and_split = runnables.RunnableLambda(fetch_doc_and_split)
    all_topic_docs = runnable_get_doc_and_split.batch(topics)

    # Flattening the list[List[Document]] -> list[Document]
    all_topic_docs: list[documents.Document] = list(itertools.chain.from_iterable(all_topic_docs))
    for doc in all_topic_docs:
        doc.page_content = "search_document: "+doc.page_content
    return all_topic_docs


'''
For given query - 
1) Fetch atmost 5 relevant wikipedia topics
2) For the given topics - Fetch and then Split [Parallel Execution]
3) Return a list of all the Documents retrieved from Wikipedia
'''
def get_all_wiki_docs(model: ChatOpenAI, search_query: str) -> list[documents.Document]:
    wiki_chain = fetch_wiki_topics(model) | get_all_topics_docs
    all_topic_docs = wiki_chain.invoke(search_query)
    return all_topic_docs


# model = ChatOpenAI(
#             api_key="api_key",
#             base_url="http://localhost:1234/v1/",
#             temperature=0.1,
#         )
# all_docs = get_all_wiki_docs(model, "The fall of USSR")
# print(all_docs)