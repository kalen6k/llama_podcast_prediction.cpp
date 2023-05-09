import json
import re
import openai
from typing import List
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

num_words = 5113
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
      api_key=openai.api_key,
      model_name="text-embedding-ada-002"
      )

client = chromadb.Client(Settings(
      chroma_db_impl="duckdb+parquet",
      persist_directory="chromadb"
))
client.persist()

# calls to the gpt chat endpoint for hypothetical document generation 
# (~ $0.014 with gpt-3.5-turbo for the entire transcript under rates for 4/2023)
# 5113 words * 1000 tokens/750 words * $0.002/1000 tokens
# For HyDE change the model to gpt-4 if transcript is longer than mine / need exact relevance among more embeddings

#calls to the gpt chat endpoint for document summarization)
# gpt-4 is $0.03/1K tokens for prompt and $0.06/1K tokens for completion
# each summary is around 200 words prompt and 100 words completion
# from dialog_summary_manage, the total number of calls is 3n at most where n is the number of chunks in the transcript
# 3n * 200 words * 1000 tokens/750 words * $0.03/1000 tokens = $0.024n for prompt
# 3n * 100 words * 1000 tokens/750 words * $0.06/1000 tokens = $0.024n for completion
# summarization is $0.048n for the entire transcript under rates for 4/2023
# or about $2.50 for the provided transcript and $15 for all cases shown here
def gptchat(
    system: str,
    window: str,
    model : str = "gpt-3.5-turbo"
    ):
  completion = openai.ChatCompletion.create(
      model = model,
      messages = [
          {"role": "system", "content": system},
          {"role": "user", "content": window}
      ], 
      temperature = 0
    )
  return completion["choices"][0]["message"]["content"]

# converts the entire text into chunks for retrieval
# not a "live" function (splits future text too)
# could also include the ability to incorporate overlap between chunks
def text_chunker(text: str, chunk_size: int = 100):
  words = text.split()
  num_words = len(words)
  num_chunks = (num_words + chunk_size - 1) // chunk_size
  chunks = [' '.join(words[i * chunk_size:(i + 1) * chunk_size]) for i in range(num_chunks)]
  # Generating start indices for each chunk
  starts = [i * chunk_size for i in range(num_chunks)]
  return chunks, starts

# replaces newlines with spaces
def cleaned_copy(chunk: str):
  return re.sub(r'\n+', ' ', chunk)

# generates an embeddings db using the transcript chunks with associated transcript index
# in order to only perform retrieval on "past" embeddings
# uses the openai embedding function
# text-embedding-ada-002 is their best model for retrieval and costs $0.0004/1000 tokens
# 5113 words * 1000 tokens/750 words * $0.0004/1000 tokens = $0.0027 for the entire transcript once
def generate_embeddings_db(
    text: str,
    db_name: str,
    chunk_size: int = 100,
    ):
  chunks, starts = text_chunker(text, chunk_size)
  # Clean the speaker chunks of newline chars before creating embeddings
  cleaned_chunks = [cleaned_copy(chunk) for chunk in chunks]
  start_ids = [str(x) for x in starts]
  collection = client.create_collection(name=db_name, embedding_function=openai_ef)
  start_indices = [{"start_index": idx} for idx in starts]
  collection.add(documents=cleaned_chunks, metadatas=start_indices, ids=start_ids)
  
  print(f"{db_name} has been stored locally")

# see gptchat for cost analysis
def summarize(chunks: List, chunks_size: int):
  # I have found gpt4 to be better at summarization
  model = "gpt-4"
  system_message = f"combine and summarize the following chunks of text in around {chunks_size} words"
  # join the two chunks with a newline in between
  user_message = "\n".join(chunks)
  return gptchat(system_message, user_message, model)

# see gptchat for cost analysis
class SummaryManager:
  def __init__(self):
    self.summary_chunks = []
    self.summary_chunks_depth = []
    self.current_summary_index = 0
    self.current_summary = ""
  def update_current_summary(self, chunk_size):
    i = len(self.summary_chunks) - 1
    merging_summary_chunks = self.summary_chunks.copy()
    while i > 0:
      #print(f"merging chunks")
      merging_summary_chunks[i - 1] = summarize([merging_summary_chunks[i - 1], merging_summary_chunks[i]], chunk_size)
      #print(merging_summary_chunks[i - 1])
      i -=1
    self.current_summary = merging_summary_chunks[0]
    #print(self.current_summary)
  def dialog_summary_manage(self, db_name, chunk_size, current_index):
    chunk_ids = [i for i in range(0, current_index, chunk_size)]
    chunks_after_sum = [chunk for chunk in chunk_ids if chunk > self.current_summary_index]
    if len(chunks_after_sum) == 2:
      collection = client.get_collection(name=db_name, embedding_function=openai_ef)
      string_ids = [str(chunk_start) for chunk_start in chunks_after_sum]
      current_chunks = collection.get(include=["documents"], ids=string_ids)['documents']
      
      #print(current_chunks)
      new_summary = summarize(current_chunks, chunk_size)
      #print(f"new summary:{new_summary}")

      # merge the new summary with the existing summary_chunks
      self.summary_chunks.append(new_summary)
      self.current_summary_index = current_index
      self.summary_chunks_depth.append(0)

      i = len(self.summary_chunks) - 1
      # num of calls to the summarization endpoint is n - 1 for the entire transcript
      # where n is the number of chunks in the transcript
      # Summarization Tree Characteristics:
      # 1. Branching factor: 2. Depth: log2(n), where n is the number of documents
      # - Total branches: 2n - 1 , Leaf nodes (original documents): n
      # - Non-leaf nodes (intermediate summaries): n - 1
      while i > 0 and self.summary_chunks_depth[i] == self.summary_chunks_depth[i-1]:
        merged_summary = summarize([self.summary_chunks[i - 1], self.summary_chunks[i]], chunk_size)
        self.summary_chunks[i - 1] = merged_summary
        self.summary_chunks_depth[i - 1] += 1
        self.summary_chunks.pop(i)
        self.summary_chunks_depth.pop(i)
        i-= 1
      # current summary is updated n/2 times
      # 2-3 summarizations occur per update_current_summary call
      #print("updating_current_summary")
      self.update_current_summary(chunk_size)

# embedding retrieval consisting of simply embedding the current window in order to retrieve relevant info
def base_embedding_retrieval(
    window: List,
    db_name: str,
    current_index: int,
    chunk_size: int,
    num_results: int
    ):
  collection = client.get_collection(name=db_name, embedding_function=openai_ef)
  if current_index < (chunk_size * 2):
    return ""
  query_text = window
  results = collection.query(
      query_texts=query_text,
      include=["documents"],
      n_results=num_results,
      # only retrieve from the "past"
      where={"start_index": {"$lt": current_index}}
  )
  return results['documents'][0]

# embedding retrieval consisting of embedding a hypothetical continuation of the dialogue in order to retrieve relevant info
def hypothetical_document_embedding_retrieval(
    window: List,
    db_name: str,
    current_index: int,
    chunk_size: int,
    num_results: int
    ):
  window = window[0]
  hypothetical_doc_instruction = f"continue writing the next {chunk_size} words for the podcast dialogue after this segment:"
  #print(hypothetical_doc_instruction + window)
  hypothetical_doc_query = gptchat(hypothetical_doc_instruction, window)
  print(hypothetical_doc_query)
  query_text = [hypothetical_doc_query]
  collection = client.get_collection(name=db_name, embedding_function=openai_ef)
  results = collection.query(
      query_texts=query_text,
      include=["documents"],
      n_results=num_results,
      # only retrieve from the "past"
      where={"start_index": {"$lt": current_index}})
  return results['documents'][0]

# embedding retrieval consisting of embedding a hypothetical continuation of the dialogue 
# created with the dialogue summary in order to retrieve relevant info
def hypothetical_document_embedding_summary_retrieval(
    window: List,
    summary: str,
    db_name: str,
    current_index: int,
    chunk_size: int,
    num_results: int
    ):
  window = window[0]
  hypothetical_doc_summary_instruction = f"use the dialog summary and the current segment to continue writing the next {chunk_size} words for the dialogue:"
  hypothetical_doc_summary_query = gptchat(hypothetical_doc_summary_instruction, "dialog summary:" + "\n" + summary + "\n" + "current segment:" + "\n" + window)
  query_text = [hypothetical_doc_summary_query]
  collection = client.get_collection(name=db_name, embedding_function=openai_ef)
  results = collection.query(
      query_texts=query_text,
      n_results=num_results,
      include=["documents"],
      # only retrieve from the "past"
      where={"start_index": {"$lt": current_index}})
  return results['documents'][0]

# function to manage the retrieval/dialog techniques
def prepend_prompt(
    prompt_type: str,
    window: List,
    chunk_size: int,
    db_name: str,
    current_index: int,
    num_results: int,
    history: SummaryManager
    ):
  if prompt_type == "original":
    return ""
  elif prompt_type == "dialog_summary":
    history.dialog_summary_manage(db_name, chunk_size, current_index)
    if history.current_summary == "":
      return ""
    instructions = "Resume the podcast script using the dialog summary provided above as a reference:"
    return history.current_summary + "\n" + instructions + "\n"
  elif prompt_type == "base_retrieval":
    if current_index < (chunk_size * 2):
      return ""
    retrieved_text = base_embedding_retrieval(window, db_name, current_index, chunk_size, num_results)
    retrieved_strings = "\n".join(retrieved_text)
    instructions = "Resume the podcast script using the retrieved text provided above as a reference:"
    return "Retrieved Text: " + retrieved_strings + "\n" + instructions + "\n"
  elif prompt_type == "hyde_retrieval":
    if current_index < (chunk_size * 2):
      return ""
    instructions = "Resume the podcast script using the retrieved text provided above as a reference:"
    retrieved_text = hypothetical_document_embedding_retrieval(window, db_name, current_index, chunk_size, num_results)
    retrieved_strings = "\n".join(retrieved_text)
    print(retrieved_strings)
    return "Retrieved Text: " + retrieved_strings + "\n" + instructions + "\n"
  elif prompt_type == "retrieval_dialog_summary":
    history.dialog_summary_manage(db_name, chunk_size, current_index)
    if history.current_summary == "":
      return ""
    retrieved_text = base_embedding_retrieval(window, db_name, current_index, chunk_size, num_results)
    retrieved_strings = "\n".join(retrieved_text)
    instructions = "Using the provided summary and retrieved text, continue the podcast script"
    return "Summary: " + history.current_summary + "\n" + "Retrieved Text: " + retrieved_strings + "\n" + instructions + "\n"
  elif prompt_type == "hyde_retrieval_dialog_summary":
    history.dialog_summary_manage(db_name, chunk_size, current_index)
    if history.current_summary == "":
      return ""
    retrieved_text = hypothetical_document_embedding_retrieval(window, db_name, current_index, chunk_size, num_results)
    retrieved_strings = "\n".join(retrieved_text)
    instructions = "Using the provided summary and retrieved text, continue the podcast script"
    return "Summary: " + history.current_summary + "\n" + "Retrieved Text: " + retrieved_strings + "\n" + instructions + "\n"
  elif prompt_type == "summary_hyde_retrieval_dialog_summary":
    history.dialog_summary_manage(db_name, chunk_size, current_index)
    summary = history.current_summary
    if summary == "":
      return ""
    retrieved_text = hypothetical_document_embedding_summary_retrieval(window, summary, db_name, current_index, chunk_size, num_results)
    retrieved_strings = "\n".join(retrieved_text)
    instructions = "Using the provided summary and retrieved text, continue the podcast script"
    return "Summary: " + history.current_summary + "\n" + "Retrieved Text: " + retrieved_strings + "\n" + instructions + "\n"

# generator function to create sliding window of text
def sliding_window(text: str, window_size: int = 10):
  words = text.split()
  for i in range(len(words) - window_size - 1):
    next_word = re.sub(r"[^a-zA-Z\'\"]", '', words[i + window_size])
    yield i, [' '.join(words[i:i + window_size])], " " + next_word
  

def generate_prompts_and_retrieved_texts(text: str, chunk_size: int, window_size: int, prompt_type: str, db_name: str, num_results: int):
  prompts = []
  history = SummaryManager()
  # slow for loop with the generator function as the memory management has to remain sequential for each json file
  for i, window, next_word in sliding_window(text, window_size=window_size):
    #if i > chunk_size:
      #print(i)
    prepend_context = prepend_prompt(prompt_type, window, chunk_size, db_name, i, num_results, history)
    prompt = {
          "prompt": prepend_context + window[0],
          "next_word": next_word
      }
    prompts.append(prompt)
  return prompts

def create_input_json(prompts, file_name):
  input_data = {
      "input_data": prompts
  }
  with open(file_name, 'w') as f:
      json.dump(input_data, f, ensure_ascii=False, indent=2)

def process_prompt_type(prompt_type: str, transcript: str, chunk_size: int, window_size: int, db_name_prefix: str, num_results: int):
  db_name = f"{db_name_prefix}_{prompt_type}_{window_size}"
  generate_embeddings_db(transcript, db_name, chunk_size)

  print(f"{prompt_type}_{window_size} generation begins")
  prompts = generate_prompts_and_retrieved_texts(transcript, chunk_size, window_size, prompt_type, db_name, num_results)

  # create input file
  file_name = f"input_{prompt_type}_{window_size}.json"
  file_path = "podcast_testing_in/"
  create_input_json(prompts, file_path + file_name)
  print(f"input_{prompt_type}_{window_size}.json created")
  

def main():
  # read the text file containing the transcript from the content directory
  with open("podcast-transcription.txt", "r") as f:
    transcript = f.read()

  # trim the transcript to the first 500 words
  #transcript = " ".join(transcript.split()[:500])

  # create embeddings db from transcript
  db_name_prefix = "podcast_embeddings"
  chunk_size = 50
  window_size = 100
  num_results = 1
  process_prompt_type("original", transcript, chunk_size, db_name_prefix, num_results)
  # using the original prompt-type create input files for window sizes from 1 to 150
  #for window_size in range(1, 151):
    #process_prompt_type("original", transcript, chunk_size, window_size, db_name_prefix, num_results)

  #process_prompt_type("hyde_retrieval", transcript, chunk_size, db_name_prefix, num_results)
  #prompt_types = ["original", "dialog_summary", "base_retrieval", "hyde_retrieval", "retrieval_dialog_summary", "hyde_retrieval_dialog_summary"]
  #with ThreadPoolExecutor() as executor:
    #executor.map(process_prompt_type, prompt_types)

if __name__ == "__main__":
  main()
