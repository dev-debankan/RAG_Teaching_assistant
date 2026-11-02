import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import os

OLLAMA_URL = "http://localhost:11434/api/embed"

def create_embedding(text_list):
    embeddings = []
    for start in range(0, len(text_list), 32):
        batch = text_list[start:start + 32]
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": "bge-m3", "input": batch},
                timeout=180,
            )
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Ollama is not reachable at http://localhost:11434. "
                "Start it with: ollama serve"
            ) from exc

        if not response.ok:
            raise RuntimeError(
                f"Ollama embedding request failed ({response.status_code}): "
                f"{response.text}"
            )

        batch_embeddings = response.json().get("embeddings", [])
        if len(batch_embeddings) != len(batch):
            raise ValueError(
                f"Ollama returned {len(batch_embeddings)} embeddings for "
                f"{len(batch)} inputs"
            )
        embeddings.extend(batch_embeddings)

    return embeddings


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model":"llama3.2",
        "prompt":prompt,
        "stream":False
    })
    if not r.ok:
        raise RuntimeError(
            f"Ollama generation request failed ({r.status_code}): {r.text}"
        )

    response = r.json()
    generated_text = response.get("response")
    if generated_text is None:
        raise ValueError(f"Ollama response did not contain 'response': {response}")
    return generated_text



df= joblib.load(os.path.join(os.path.dirname(__file__), "embeddings.joblib"))  ##load the df from the pickle file




incoming_query=input("Enter your query: ")  ##take input from the user
question_embedding = create_embedding([incoming_query])[0]  ##create embedding for the incoming query

##now we will calculate cosine similarity of question embeddings 

similarities=cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten() ##calculate cosine similarity between the embeddings of the chunks and the question embedding
#print(similarities)  ##print the similarities
# print (np.vstack(df['embedding'].values))
# print (np.vstack(df['embedding'].shape))

similarities.argsort()
top_results=7
max_idx=similarities.argsort()[::-1][0:top_results]  ##get the indices of the top k most similar chunks
#print(max_idx)
new_df=df.loc[max_idx]  ##get the top k most similar chunks
#print(new_df)  ##print the top k most similar chunks
#print(new_df[["title","text","number"]])  ##print the text of the top k most similar chunks

prompt = f'''I am teaching Compiler designing. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, and content at that time:

{new_df[["title","number",'start','end',"text"]].to_json(orient="records",lines=True)}

---------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way(dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)  ##write the prompt to a text file

response=inference(prompt)  ##call the inference function to get the response from the model
print(response)

with open(os.path.join(os.path.dirname(__file__), "response.txt"), "w", encoding="utf-8") as f:
    f.write(response)  ##write the generated text to a text file


# for index, item in new_df.iterrows():
#     print(index,item['title'],item['text'],item['number'])  ##print the text of the top k most similar chunks