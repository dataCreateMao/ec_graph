from utils import MysqlReader,Neo4jWriter

class TableSync:
    def __init__(self):
        self.reader = MysqlReader()
        self.writer = Neo4jWriter()

    def sync_category1(self):
        sql = """
        select * 
        from base_category1
        """
        category1 = self.reader.read(sql)
        self.writer.write_nodes('Category1', category1)

    def sync_category2(self):
        sql = """
        select * 
        from base_category2
        """
        category2 = self.reader.read(sql)
        self.writer.write_nodes('Category2', category2)

    def sync_category3(self):
        sql = """
        select * 
        from base_category3
        """
        category3 = self.reader.read(sql)
        self.writer.write_nodes('Category3', category3)

    # 从下级分类表中，提取与上级的关系
    def sync_category2_to_category1(self):
        sql = """
        select c2.id as start_id, c2.category1_id as end_id
        from base_category2 c2 
        """
        relations = self.reader.read(sql)
        self.writer.write_relations('Belong', 'Category2', 'Category1', relations)

    def sync_category3_to_category2(self):
        sql = """
        select c3.id as start_id, c3.category2_id as end_id
        from base_category3 c3 
        """
        relations = self.reader.read(sql)
        self.writer.write_relations('Belong', 'Category2', 'Category1', relations)

    def sync_base_attr_name(self):
        sql = """
        select id, attr_name as name
        from base_attr_info
        """
        attr_names = self.reader.read(sql)
        self.writer.write_nodes('BaseAttrName', attr_names)

    def sync_base_attr_value(self):
        sql = """
        select id, value_name as name
        from base_attr_value
        """
        attr_values = self.reader.read(sql)
        self.writer.write_nodes('BaseAttrValue', attr_values)

    def sync_base_attr_name_to_value(self):
        sql = """
        select a.id as end_id, a.attr_id as start_id
        from base_attr_value a
        """
        relations = self.reader.read(sql)
        self.writer.write_relations('Have', 'BaseAttrName', 'BaseAttrValue', relations)

    def sync_category1_to_base_attr_name(self):
        sql = """
        select a.category_id as start_id, a.id as end_id
        from base_attr_info a
        where a.category_level = 1
        """
        relations = self.reader.read(sql)
        self.writer.write_relations('Have', 'Category1', 'BaseAttrName', relations)

    def sync_category2_to_base_attr_name(self):
        sql = """
        select a.category_id as start_id, a.id as end_id
        from base_attr_info a
        where a.category_level = 2
        """
        relations = self.reader.read(sql)
        self.writer.write_relations('Have', 'Category1', 'BaseAttrName', relations)

    def sync_category3_to_base_attr_name(self):
        sql = """
        select a.category_id as start_id, a.id as end_id
        from base_attr_info a
        where a.category_level = 3
        """
        relations = self.reader.read(sql)
        self.writer.write_relations('Have', 'Category1', 'BaseAttrName', relations)

    #商品信息
    def sync_spu(self):
        sql = """
        select id, spu_name as name
        from spu_info
        """
        spus = self.reader.read(sql)
        self.writer.write_nodes('SPU', spus)

    def sync_sku(self):
        sql = """
        select id, sku_name as name
        from sku_info
        """
        skus = self.reader.read(sql)
        self.writer.write_nodes('SKU', skus)

    def sync_sku_to_spu(self):
        sql = """
        select s.id as start_id, s.spu_id as end_id
        from sku_info s
        """
        relations = self.reader.read(sql)
        self.writer.write_relations('Belong', 'SKU', 'SPU', relations)

    def sync_spu_to_category3(self):
        sql = """
            select 
                id as start_id,
                category3_id as end_id
            from spu_info
        """
        relations = self.reader.read(sql)
        self.writer.write_relations('Belong', 'SPU', 'Category3', relations)

    def sync_trademark(self):
        sql = """
        select 
            id, tm_name as name
        from base_trademark
        """
        trademarks = self.reader.read(sql)
        self.writer.write_nodes('Trademark', trademarks)

    def sync_spu_to_trademark(self):
        sql = """
        select 
            id as start_id,
            tm_id as end_id
        from spu_info
        """
        relations = self.reader.read(sql)
        self.writer.write_relations('Belong', 'SKU', 'Trademark', relations)

    def sync_sale_attr_name(self):
        sql = """
            select id,sale_attr_name  as name
            from spu_sale_attr
        """
        sale_attr_names = self.reader.read(sql)
        self.writer.write_nodes('SaleAttrName', sale_attr_names)

    def sync_sale_attr_value(self):
        sql = """
            select 
                id,
                sale_attr_value_name as name
            from spu_sale_attr_value
        """
        sale_attr_values = self.reader.read(sql)
        self.writer.write_nodes('SaleAttrValue', sale_attr_values)

    def sync_sale_attr_name_to_value(self):
        sql = """
            select 
                a.id as start_id,
                v.id as end_id
            from spu_sale_attr a 
            join spu_sale_attr_value v
            on a.spu_id = v.spu_id and a.base_sale_attr_id = v.base_sale_attr_id
        """
        relations = self.reader.read(sql)
        self.writer.write_relations('Have', 'SaleAttrName', 'SaleAttrValue', relations)

    def sync_spu_to_sale_attr_name(self):
        sql = """
            select 
                a.spu_id as start_id,
                a.id as end_id
            from spu_sale_attr a
        """
        relations = self.reader.read(sql)
        self.writer.write_relations('Have', 'SPU', 'SaleAttrName', relations)

    def sync_sku_to_base_attr_value(self):
        sql = """
                select 
                    a.sku_id as start_id,
                    a.value_id as end_id
                from sku_attr_value a 
            """
        relations = self.reader.read(sql)
        self.writer.write_relations('Have', 'SKU', 'SaleAttrValue', relations)

    def sync_sku_to_sale_attr_value(self):
        sql = """
            select 
                a.sku_id as start_id,
                a.sale_attr_value_id as end_id
            from sku_sale_attr_value a 
        """
        relations = self.reader.read(sql)
        self.writer.write_relations('Have', 'SKU', 'BaseAttrValue', relations)

if __name__ == '__main__':
    ts = TableSync()
    ts.sync_category1()
    ts.sync_category2()
    ts.sync_category3()
    ts.sync_category2_to_category1()
    ts.sync_category3_to_category2()
    ts.sync_base_attr_name()
    ts.sync_base_attr_value()
    ts.sync_base_attr_name_to_value()
    ts.sync_category1_to_base_attr_name()
    ts.sync_category2_to_base_attr_name()
    ts.sync_category3_to_base_attr_name()
    ts.sync_spu()
    ts.sync_sku()
    ts.sync_sku_to_spu()
    ts.sync_spu_to_category3()
    ts.sync_trademark()
    ts.sync_spu_to_trademark()
    ts.sync_sale_attr_name()
    ts.sync_sale_attr_value()
    ts.sync_sale_attr_name_to_value()
    ts.sync_spu_to_sale_attr_name()
    ts.sync_sku_to_sale_attr_value()
    ts.sync_sku_to_base_attr_value()



