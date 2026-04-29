import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType

# 兼容 `python src/web/service.py` 直接启动方式，确保可导入项目内包
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configuration.config import *
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
class ChatService:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG["uri"],
            username=NEO4J_CONFIG["auth"][0],
            password=NEO4J_CONFIG["auth"][1]
        )
        self.embedding_model = None
        load_dotenv()
        # LLM 
        self.llm = ChatOpenAI(
            model="deepseek-v4-pro",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0.7,
        )
        self.fulltext_indexes = {
            "Trademark": "trademark_fulltext_index",
            "SPU": "spu_fulltext_index",
            "SKU": "sku_fulltext_index",
            "Category1": "category1_fulltext_index",
            "Category2": "category2_fulltext_index",
            "Category3": "category3_fulltext_index",
        }
        # 默认使用混合检索；若本机 torch 与 numpy 不兼容，则退化为全文检索兜底
        self.neo4j_vectors = {}
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                # model_name="BAAI/bge-large-zh-v1.5",
                encode_kwargs={"normalize_embeddings": True}
            )
            self.neo4j_vectors = {
                "Trademark": Neo4jVector.from_existing_index(
                    index_name="trademark_vector_index",
                    keyword_index_name="trademark_fulltext_index",
                    embedding=self.embedding_model,
                    url=NEO4J_CONFIG["uri"],
                    username=NEO4J_CONFIG["auth"][0],
                    password=NEO4J_CONFIG["auth"][1],
                    search_type=SearchType.HYBRID,
                ),
                "SPU": Neo4jVector.from_existing_index(
                    index_name="spu_vector_index",
                    keyword_index_name="spu_fulltext_index",
                    embedding=self.embedding_model,
                    url=NEO4J_CONFIG["uri"],
                    username=NEO4J_CONFIG["auth"][0],
                    password=NEO4J_CONFIG["auth"][1],
                    search_type=SearchType.HYBRID,
                ),
                "SKU": Neo4jVector.from_existing_index(
                    index_name="sku_vector_index",
                    keyword_index_name="sku_fulltext_index",
                    embedding=self.embedding_model,
                    url=NEO4J_CONFIG["uri"],
                    username=NEO4J_CONFIG["auth"][0],
                    password=NEO4J_CONFIG["auth"][1],
                    search_type=SearchType.HYBRID,
                ),
                "Category1": Neo4jVector.from_existing_index(
                    index_name="category1_vector_index",
                    keyword_index_name="category1_fulltext_index",
                    embedding=self.embedding_model,
                    url=NEO4J_CONFIG["uri"],
                    username=NEO4J_CONFIG["auth"][0],
                    password=NEO4J_CONFIG["auth"][1],
                    search_type=SearchType.HYBRID,
                ),
                "Category2": Neo4jVector.from_existing_index(
                    index_name="category2_vector_index",
                    keyword_index_name="category2_fulltext_index",
                    embedding=self.embedding_model,
                    url=NEO4J_CONFIG["uri"],
                    username=NEO4J_CONFIG["auth"][0],
                    password=NEO4J_CONFIG["auth"][1],
                    search_type=SearchType.HYBRID,
                ),
                "Category3": Neo4jVector.from_existing_index(
                    index_name="category3_vector_index",
                    keyword_index_name="category3_fulltext_index",
                    embedding=self.embedding_model,
                    url=NEO4J_CONFIG["uri"],
                    username=NEO4J_CONFIG["auth"][0],
                    password=NEO4J_CONFIG["auth"][1],
                    search_type=SearchType.HYBRID,
                ),
            }
        except Exception as exc:
            print(f"[WARN] 向量检索初始化失败，降级为全文检索：{exc}")
        # 定义Parser
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()
    
    # 核心聊天服务流程
    def chat(self, question: str) -> str:
        # 1. 根据用户问题，生成Cypher以及需要对齐的实体
        result = self._generate_cypher(question, self.graph.schema)
        cypher = result["cypher_query"]
        entities_to_align = result["entities_to_align"]
        print(cypher)
        print("对齐之前的实体：",entities_to_align)
        
        # 2.实体对齐（混合检索）
        aligned_entities = self._align_entities(entities_to_align)
        print("对齐之后的实体：",aligned_entities)
        
        # 3.执行Cypher，获取结果
        query_result = self._execute_cypher(cypher, aligned_entities)
        print("查询结果：",query_result)
        
        # 4.根据用户问题和查询结果生成答案
        answer = self._generate_answer(question, query_result)
        print("最终答案：",answer)
        return answer
        

    # 1. 根据用户问题，调用LLM生成Cypher以及需要对齐的实体
    def _generate_cypher(self, question: str, schema_info: str) -> dict:
        prompt = f"""
        你是一个专业的Neo4j Cypher查询生成器。你的任务是根据用户问题生成一条Cypher查询语句，用于从知识图谱中获取回答用户问题所需的信息。

        用户问题：{question}

        知识图谱结构信息：{schema_info}

        要求：
        1. 生成参数化Cypher查询语句,用param_0, param_1等代替具体值,避免使用具体的值
        2. 识别需要对齐的实体
        3. 必须严格使用以下JSON格式输出结果
        {{
         "cypher_query": "生成的Cypher语句",
         "entities_to_align": [
          {{
           "param_name": "param_0",
           "entity": "原始实体名称",
           "label": "节点类型"
          }}
         ]
        }}"""
        result = self.llm.invoke(prompt)
        return self.json_parser.parse(result.content)
    
    # 2. 实体对齐（混合检索）
    def _align_entities(self, entities_to_align):
        for index, entity in enumerate(entities_to_align):
            label = entity["label"]
            entity = entity["entity"]
            if label in self.neo4j_vectors:
                aligned_entity = self.neo4j_vectors[label].similarity_search(entity, k=1)[0].page_content
            else:
                fulltext_index = self.fulltext_indexes.get(label)
                aligned_entity = entity
                if fulltext_index:
                    records = self.graph.query(
                        """
                        CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score
                        RETURN coalesce(node.name, node.title, node.id, toString(node)) AS value
                        ORDER BY score DESC
                        LIMIT 1
                        """,
                        params={"index_name": fulltext_index, "query": entity},
                    )
                    if records and records[0].get("value"):
                        aligned_entity = records[0]["value"]
            entities_to_align[index]["entity"] = aligned_entity
        return entities_to_align
    
    # 3.用对齐的实体名称替换param_0，执行Cypher，获取结果
    def _execute_cypher(self, cypher_query, aligned_entities):
        # 提取对齐后的实体名称
        params = {aligned_entity["param_name"]: aligned_entity["entity"] for aligned_entity in aligned_entities}
        # 执行Cypher
        return self.graph.query(cypher_query, params=params)

    # 4.根据用户问题和查询结果生成答案
    def _generate_answer(self, question, query_result):
        template = f"""
        你是一个电商智能客服，根据用户问题，以及数据库查询结果生成一段简洁、准确的自然语言回答。
        用户问题: {question}
        数据库返回结果: {query_result}
        """
        prompt = template.format(question=question, query_result=query_result)
        result = self.llm.invoke(prompt)
        return self.str_parser.parse(result.content)

if __name__ == "__main__":
    chat_service = ChatService()
    chat_service.chat("Apple有哪些产品？")