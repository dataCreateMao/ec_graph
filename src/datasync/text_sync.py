import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification

from configuration.config import *
from utils import MysqlReader,Neo4jWriter
from ner.predict import Predictor

class TextSynchronizer:
    def __init__(self):
        self.mysql_reader = MysqlReader()
        self.neo4j_writer = Neo4jWriter()
        # 定义一个实体的提取器，本质就是Predictor
        self.predictor = self._init_extractor()

    # 初始化一个Predictor
    def _init_extractor(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / 'best_model'))
        model = AutoModelForTokenClassification.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / 'best_model'))
        return Predictor(model, tokenizer, device)

    # 同步TAG 标签
    def sync_tag(self):
        # 1. 从MySQL提取商品描述信息
        sql = """
            select id, description
            from spu_info
        """
        spu_descs = self.mysql_reader.read(sql)
        # 2. 拆分spu id 和 desc
        ids = [item['id'] for item in spu_descs]
        descs = [item['description'] for item in spu_descs]

        # 3. 提取所有数据的 Tag 列表
        tags_list = self.predictor.predict(descs)

        # 4. 构建 Tag 节点的属性(id, name)，以及 SPU -> Tag 的关系(start_id, ennd_id)
        tag_properties = []
        relations = []
        for id, tags in zip(ids, tags_list):
            # 遍历当前SPU的所有Tag
            for index,tag in enumerate(tags):
                tag_id = '-'.join([str(id), str(index)])
                pt = {
                    'id': tag_id,
                    'name': tag
                }
                tag_properties.append(pt)

                # 构建关系
                relation = {
                    'start_id': id,
                    'end_id': tag_id
                }
                relations.append(relation)

        # 5. 写入关系
        self.neo4j_writer.write_nodes('Tag', tag_properties)
        self.neo4j_writer.write_relations('Have', 'SPU', 'Tag', relations)


