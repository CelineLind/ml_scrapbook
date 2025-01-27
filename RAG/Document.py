from chromadb.utils import embedding_functions # type: ignore

class Document:
    def __init__(self, filepath):
        self.filepath  = filepath
        self.fulltext  = self.readFullText()
        self.embedding = self.createEmbedding()

    def readFullText(self):
        f = open(self.filepath, "r")
        fulltext = f.read()
        f.close()
        return fulltext

    def getText(self):
        return self.fulltext
    
    def createEmbedding(self):
        embed_func = embedding_functions.DefaultEmbeddingFunction()
        return embed_func([self.fulltext])
