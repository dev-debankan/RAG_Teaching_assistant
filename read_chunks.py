import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib


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

# a = create_embedding("Hello, world!i am debankan")
# print(a)

jsons=os.listdir("jsons")    ##list all the json files in the jsons folder
# print(jsons) 
my_dicts=[]
chunk_id=0



for json_file in jsons :
    with open(f"jsons/{json_file}", "r") as f:
        content=json.load(f)
    print(f"creating embeddings for {json_file}")   
    embeddings=create_embedding([c['text'] for c in content['chunks']]) ##listed all the embeddings and passed in create_embedding function
   
    for i,chunk in enumerate(content['chunks']) :
     
        chunk['chunk_id'] = chunk_id
        
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
        # print(chunk)
   
        
 



# print(my_dicts)

# a=create_embedding(["Hello, world! I am Debankan"])
# print(a)

df=pd.DataFrame.from_records(my_dicts)  ##convert the list of dictionaries to a dataframe
'''Save the embeddings into a df permanantly using job lib '''
joblib.dump(df, os.path.join(os.path.dirname(__file__), "embeddings.joblib"))  ##save the dataframe to a joblib file
# print(df)
