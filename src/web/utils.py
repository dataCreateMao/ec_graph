from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from neo4j_graphrag.types import SearchType

from configuration.config import *
from langchain_huggingface import HuggingFaceEmbeddings


class IndexUtils:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG["uri"],
            username=NEO4J_CONFIG["auth"][0],
            password=NEO4J_CONFIG["auth"][1]
        )
        # 定义模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            # model_name="BAAI/bge-large-zh-v1.5",
            encode_kwargs={"normalize_embeddings": True}
        )

    # 创建全文索引,传入索引名称、节点标签、属性
    def create_fulltext_index(self, index_name, node_label, property_name):
        cypher = f"""
            CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
            FOR (n:{node_label}) ON (n.{property_name})
            """
        self.graph.query(cypher)

    # 创建向量索引,需要传入生成向量的“源属性”，以及嵌入向量属性
    def create_vector_index(self, index_name, node_label, source_property, embedding_property):
        # 生成嵌入向量，并添加到节点属性中
        embedding_dim = self._add_embedding(node_label, source_property, embedding_property)
        cypher = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{node_label}) ON (n.{embedding_property})
            OPTIONS {{
                "indexConfig": {{
                    "spaceType": "cosine",
                    "dimensions": {embedding_dim}
                }}
            }}
            """
        self.graph.query(cypher)

    # 内部函数：生成嵌入向量，并添加到节点属性中，返回向量维度
    def _add_embedding(self, node_label, source_property, embedding_property):
        # 1. 查询所有节点类型对应的源属性值，作为模型的输入;还需要查出节点id
        cypher = f"""
            MATCH (n:{node_label})
            RETURN id(n) AS id, n.{source_property} AS text
        """
        results = self.graph.query(cypher)
        # 2. 获取查询结果中的文本内容
        docs = [result["text"] for result in results]
        # 3. 调用嵌入模型，得到嵌入向量
        embeddings = self.embedding_model.embed_documents(docs)
        # 4. 将id和嵌入向量组合成字典形式
        batch = [{"id": result["id"], embedding_property: embedding} for result, embedding in zip(results, embeddings)]
        # 5. 执行cypher，按id查节点，写入新的嵌入向量属性
        cypher = f"""
            UNWIND $batch as item
            MATCH (n:{node_label}) WHERE id(n) = item.id
            SET n.{embedding_property} = item.{embedding_property}
        """
        self.graph.query(cypher, params={"batch": batch})
        return len(embeddings[0])


if __name__ == '__main__':
    index = IndexUtils()
    index.create_fulltext_index("trademark_fulltext_index", "Trademark", "name")
    index.create_vector_index("trademark_vector_index", "Trademark", "name", "embedding")

    # # 测试
    # index_name = "trademark_vector_index"
    # keyword_index_name = "trademark_keyword_index"
    # store = Neo4jVector.from_existing_index(
    #     index.embedding_model,
    #     index_name=index_name,
    #     keyword_index_name=keyword_index_name,
    #     url=NEO4J_CONFIG["uri"],
    #     username=NEO4J_CONFIG["auth"][0],
    #     password=NEO4J_CONFIG["auth"][1],
    #     search_type=SearchType.HYBRID,
    # )
    # # result = store.similarity_search("Apple", k=5)
    # result = store.similarity_search("Apple", k=1)[0].page_content
    # print(result)

    index.create_fulltext_index("spu_fulltext_index", "SPU", "name")
    index.create_vector_index("spu_vector_index", "SPU", "name", "embedding")
    index.create_fulltext_index("sku_fulltext_index", "SKU", "name")
    index.create_vector_index("sku_vector_index", "SKU", "name", "embedding")
    index.create_fulltext_index("category1_fulltext_index", "Category1", "name")
    index.create_vector_index("category1_vector_index", "Category1", "name", "embedding")
    index.create_fulltext_index("category2_fulltext_index", "Category2", "name")
    index.create_vector_index("category2_vector_index", "Category2", "name", "embedding")
    index.create_fulltext_index("category3_fulltext_index", "Category3", "name")
    index.create_vector_index("category3_vector_index", "Category3", "name", "embedding")
