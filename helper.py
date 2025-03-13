def get_system_prompt(content_list: list[str]) -> str:
    system_prompt = \
    f'''
{"\n\n".join([f"Context {i+1}: {content}" for i, content in enumerate(content_list)])}
    '''

    return system_prompt

def get_wiki_topic_template():
    template = \
    """
<|start_header_id|>system<|end_header_id|>
Context: You are a Wikipedia researcher and a helpful assistant. You list at most 5 relevant topics, including basics, to a query asked. 
Only concise searchable topic headings at most 4 words, nothing else. If you don't know the answer, just say "<unretrievableerror>".
Strictly structure the output as comma seperated values.

<|start_header_id|>user<|end_header_id|>
Query: {query}

<|start_header_id|>assistant<|end_header_id|>
    """
    return template

def get_generation_template():
    template = \
    """
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant and smart and funny journalist. You summarize contexts very well and present information as asked.
Use the following context when responding:

{context}


<|start_header_id|>user<|end_header_id|>
Write a brief article with a catchy headline according to User Query. Strictly output 1 headline and 3 other paragraphs.
User Query: {query}


<|start_header_id|>assistant<|end_header_id|>
    """
    return template