import gradio as gr
from dotenv import load_dotenv
import os
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint

#import requests

def process_conversation(conversation):
    #print("Entered Conversation ---> ", conversation)
    load_dotenv()
    hf_api_key = os.getenv("hf_token")
    model_name = os.getenv("llm_model_name")
    #llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    llm = HuggingFaceEndpoint(repo_id=model_name, max_length=128, temperature=0.5, huggingfacehub_api_token=hf_api_key)
    print("LLM created")
    tokens = llm.get_num_tokens(conversation)
    print("No. of tokens = ", tokens)
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([conversation])
    num_docs = len(docs)
    num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)
    print (f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens")
    map_prompt = """
        Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    combine_prompt = """
        Write a concise summary of the following text including both patient's concern and doctor's advice delimited by triple backquotes.
        Return your response in bullet points which covers the key points of the text.
        ```{text}```
        BULLET POINT SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
                                     #verbose=True
    )

    output = output = summary_chain.invoke(docs)
    
    return output['output_text']

with gr.Blocks(gr.themes.Soft(),title="Summarization App : Patient vs Doctor Conversation (Built on Open Source)") as demo:
    gr.Markdown(
        """
        # Summarization App : Patient vs Doctor Conversation (Built on Open Source)
        """)
    textbox = gr.Textbox(lines=1, max_lines=500,label="Enter the conversation details to summarize", value="")
    with gr.Row():
        button = gr.Button("Submit", variant="Primary")
    with gr.Column():
        output1 = gr.Textbox(lines=1, max_lines=50, label="Summarized Conversation")   
        #output2 = gr.Textbox(lines=1, max_lines=10, label="LLM output")
    button.click(process_conversation, textbox, outputs=[output1])
demo.launch(share=True)