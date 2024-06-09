# from openai import OpenAI
from langchain import OpenAI

import httpx
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, \
    ServiceContext
from IPython.display import Markdown, display


def construct_index(directory_path):

    load_dotenv(".env", override=True)
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    # llm_predictor = LLMPredictor(llm=OpenAI(
    #
    #     temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs
    # ))

    llm_predictor = LLMPredictor(llm=OpenAI(model_name="mpt-30b-instruct"))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index


def ask_ai():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while True:
        query = input("What do you want to ask? ")
        response = index.query(query)
        display(Markdown(f"Response: <b>{response.response}</b>"))


# def main():
#     load_dotenv('.env', override=True)
#
#     client = OpenAI(
#         base_url='https://opensource-llm-api.aiaccel.dell.com/v1/',
#         http_client=httpx.Client(verify=False)
#     )
#
#     streaming = True
#     max_output_tokens = 200
#
#     # Available Models list
#     available_model_apis = ["falcon-40b-instruct", "mpt-30b-instruct", "llama-2-13b-chat", "codellama-13b-instruct",
#                             "zephyr-7b-beta"]
#     # Let's select the model from available list
#     model_selected = available_model_apis[2]
#
#     completion = client.completions.create(
#         model=model_selected,
#         max_tokens=max_output_tokens,
#         prompt=f'Can you explain who are the Los Angeles Dodgers and what are they known for is in less than {max_output_tokens} tokens?',
#         stream=streaming)
#
#     if streaming:
#         for chunk in completion:
#             print(chunk.choices[0].text, end='')
#     else:
#         print(completion.choices[0].text)


def main():
    construct_index("data/")
    ask_ai()


if __name__ == "__main__":
    main()
