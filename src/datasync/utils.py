import pymysql
from neo4j import GraphDatabase
from pymysql.cursors import DictCursor

from configuration.config import *

# 读取MySQL工具类
class MysqlReader:
    def __init__(self):
        self.conn = pymysql.connect(**MYSQL_CONFIG)
        self.cursor = self.conn.cursor(DictCursor)

    # 查询数据
    def read(self, sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    # 关闭
    def close(self):
        self.cursor.close()
        self.conn.close()

class Neo4jWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(**NEO4J_CONFIG)

    #写入节点(批量，固定标签)
    def write_nodes(self, label:str, properties:list[dict]):
        cypher = f"""
        UNWIND $batch as item
        MERGE (:{label} {{id: item.id, name: item.name}} )
        """
        self.driver.execute_query(cypher, batch= properties)

    #写入关系
    def write_relations(self, type:str, start_label:str, end_label:str, relations:list[dict]):
        cypher = f"""
                    UNWIND $batch as item
                    MATCH (start:{start_label}{{id: item.start_id}}),(end:{end_label}{{id: item.end_id}})
                    MERGE (start)-[:{ type}]->(end)
                    """
        self.driver.execute_query(cypher, batch= relations)

if __name__ == '__main__':
    mysql = MysqlReader()
    writer = Neo4jWriter()
    # 定义neo4j 的driver
    driver = GraphDatabase.driver(**NEO4J_CONFIG)
    # 1. Category1
    # 1.1 读取base_category1 数据
    sql = """
    select *
    from base_category1
    """
    category1 = mysql.read(sql)
    print(category1)
    writer.write_nodes('Category1', category1)
    # # mysql.close()
    #
    # 1.2 写入neo4j，标签 category1
    # for item in category1:
    #     # cypher = """
    #     # MERGE (c:Category1 {id: $id , name: $name})
    #     # """
    #     # driver.execute_query(cypher, parameters_= item)
    # cypher = """
    #     UNWIND $category1 as item
    #     MERGE (:Category1 {id: item.id, name: item.name} )
    # """
    # driver.execute_query(cypher, category1= category1)
    #
    # 2. Category2
    # 2.1 读取base_category2 数据
    sql = """
        select *
        from base_category2
        """
    category2 = mysql.read(sql)
    print(category2)
    writer.write_nodes('Category2', category2)
    # # 2.2 写入neo4j，标签 category2
    # cypher = """
    #         UNWIND $category2 as item
    #         MERGE (:Category2 {id: item.id, name: item.name} )
    #     """
    # driver.execute_query(cypher, category2=category2)
    #
    # 3. Category2 -Belong-> Category1
    # 3.1 读取base_category1 数据
    sql = """
            select
                id as start_id,
                category1_id as end_id
            from base_category2
            """
    relations = mysql.read(sql)
    print(relations)
    writer.write_relations('Belong', 'Category2', 'Category1', relations)

    # 2.2 写入neo4j，标签 Belong
    # cypher = """
    #             UNWIND $relations as item
    #             MATCH (start:Category2{id: item.start_id}),(end:Category1{id: item.end_id})
    #             MERGE (start)-[:Belong]->(end)
    #         """
    # driver.execute_query(cypher, relations=relations)

    mysql.close()
