# 1. 加载文档
# 2. 创建索引
# 3. 创建查询引擎
# 4. 提问
import os

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, get_response_synthesizer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# 向量数据库
EMBEDDING_DIM = 1536
COLLECTION_NAME = "full_demo"
PATH = "./qdrant_db"
client = QdrantClient(path=PATH)

# 创建 LLM 和嵌入模型
llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX, api_key=os.getenv("DASHSCOPE_API_KEY"))
embedding = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1)

Settings.llm = llm
Settings.embed_model = embedding

# 指定全局文档处理的 ingestion pipeline
Settings.transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=200)]

# 1. 加载本地文档
documents = SimpleDirectoryReader(input_dir="./data").load_data()

# 如果存在，则删除
if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)

# 2. 创建collection
client.create_collection(collection_name=COLLECTION_NAME, vectors_config={"size": EMBEDDING_DIM, "distance": "Cosine"})

# 3. 创建 vector store
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)

# 4. 指定 vector store的 storage 用于 index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# 5. 定义检索后排序模型
reranker = LLMRerank(llm=llm, top_n=2)
# 最终打分低于 0.6 分的过滤掉
sp = SimilarityPostprocessor(similarity_cutoff=0.6)

# 6. 定义 rag fusion 检索器
fusion_retriever = QueryFusionRetriever(retrievers=[index.as_retriever()],
                                 similarity_top_k=5,  # 检索召回 top k 结果
                                 num_queries=3,  # 生成 query 数
                                 use_async=False)

# 7. 构建单轮 query engine
query_engine = RetrieverQueryEngine(retriever=fusion_retriever, node_postprocessors=[reranker],
                              response_synthesizer=get_response_synthesizer(response_mode=ResponseMode.REFINE))

# 8. 对话引擎
chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine=query_engine)

# 多轮对话
while True:
    question = input("请输入问题：")
    if question.strip() == "":
        break
    answer = chat_engine.chat(question)
    print(f"{answer}")
