import chromadb # type: ignore

class VectorDatabase:
    def __init__(self):
        self.database_client = None
        self.collection = None

    def setClient(self, persistent=False):
        if persistent:
            self.database_client = chromadb.PersistentClient(path="vec_db") # persistent db
        else:
            self.database_client = chromadb.Client() # in memory db

    def getClient(self):
        return self.database_client

    def createCollection(self, collection_name):
        self.collection = self.database_client.create_collection( name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )

    def getCollection(self):
        return self.collection

    def addToCollection(self, docs, embeds, meta, ids):
        self.collection.upsert(
            documents = docs,
            embeddings= embeds,
            metadatas = meta,
            ids       = ids
        )

    def queryDatabase(self, query, n_results=5):
        results = self.collection.query(
            query_texts=query,
            n_results=n_results
        )
        return results
