import chromadb
def store_in_db ( text_chunks, embedding):
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="rag_chunks")
    for i, (chunk, vector) in enumerate (zip(text_chunks, embedding)):
        collection.add(
            documents=[chunk],
            embeddings=[vector],
            ids=[str(i)]
        )
    return collection

def retrive_chunks(query, collection, embedder,top_k = 3):
    query_vector = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents"]
    )
    return results['documents'][0]