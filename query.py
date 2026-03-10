import os
from dotenv import load_dotenv
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv('config.env')
Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key = os.getenv('OPENAI_APIKEY'))
ticker = 'NVDA'
storage_context = StorageContext.from_defaults(persist_dir=f"./storage/{ticker}")
index = load_index_from_storage(storage_context)
# After: index = load_index_from_storage(storage_context)

print("=== Index Health Check ===")
print("Index type:", type(index).__name__)                     # should be VectorStoreIndex

# A. How many documents were stored?
print("Number of documents in docstore:", len(index.docstore.docs))

# B. How many nodes (chunks) are actually indexed?
#    This is the most important number - should be > 0
print("Number of nodes in vector store:",
      len(index.vector_store.get_nodes()) if hasattr(index.vector_store, 'get_nodes') else "N/A (check via .index_struct)")

# C. Quick look at index_struct (low-level)
print("Index struct keys / length:", len(index.index_struct.nodes_dict) if hasattr(index.index_struct, 'nodes_dict') else "No nodes_dict")

# D. Try to retrieve something very generic (should match almost anything)
test_nodes = index.as_retriever(similarity_top_k=3).retrieve("finance OR company OR stock")
print("Retrieved nodes count on generic query:", len(test_nodes))
if test_nodes:
    print("Sample node text preview:", test_nodes[0].node.text[:250])
    print("Score of top node:", test_nodes[0].score)
else:
    print("-> Zero nodes retrieved even on broad query <- PROBLEM HERE")

print("=========================")
