import os, os.path

from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema, Collection
import sqlite3
import ollama
import logging
import pandas as pd
import math
import beliefs
from support import Support
import params
import Levenshtein
import globals
from tenacity import *

"""
Classes:
    - BeliefStore: persistent storage and indexing for beliefs
    - BSContext: context manager wrapping the belief store
"""

log = logging.getLogger()


class BeliefStore:
    """
    This stores beliefs, supports, texts. Interface to sql database and vector
    store

    Methods:
        - get_milvus_client: initialize milvus as vector store
        - ensure_tables: make sure tables exist in the database
        - exit: serialize and close persistent data stores
        - get_embedding: return the embedding ofr a string
        - get_stmt_id: given a text, return its id
        - get_similar_stmts: given a text, return stmts similar to it
        - get_milvus_id_dist: get closest stnt id and distance from the text
        - get_similar_beliefs: find beliefs similar to the one provided
        - dump_vector_store: return string rep of vector store
        - dump_table: return string rep of a sql table
        - get_belief_row_by_id: given an id, return the belief table row
        - get_support_row_by_it: given id, return the support table row
    """
    def __init__(self,
                 milvus_file=params.milvusLoc,
                 sqlite_file=params.sqliteLoc,
                 ):
        """
        Initial setup

        Sets up the database and the vector store.
        Creates database and tables if needed.
        :param milvus_file: location of the milvus file
        :param sqlite_file: location of the sqlite file
        """
        log.info(f"BeliefStore.init {milvus_file}, {sqlite_file}")
        self.mv_client = self.get_milvus_client(milvus_file)
        self.conn = sqlite3.connect(sqlite_file)
        self.ensure_tables()
        self.bset_set = set()
        globals.bstore = self
        log.debug(f"done bstore construction")

    def get_milvus_client(self, milvus_file):
        """
        Gets the milvus client
        :param milvus_file:
        :return: the milvus client
        """
        log.info("BeliefStore.get_milvus_client")
        mv_client = MilvusClient(milvus_file)
        id_fs = FieldSchema(name="stmt_id", dtype=DataType.INT64, is_primary=True,
                            description='Id of the statement')
        embedding_fs = FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR,
                                   dim=params.EMBEDDING_DIM, description='Embedding')
        collection_schema = CollectionSchema(fields=[id_fs, embedding_fs], auto_id=False)

        index_params = mv_client.prepare_index_params()
        index_params.add_index(field_name="stmt_id", index_type="")
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX",
                               metric_type=params.EMBEDDING_METRIC)
        if params.EMBEDDING_METRIC == 'L2':
            self.range_filter = lambda x: {"radius": x, "range_filter": 0}
        else:  # COSINE, IP
            self.range_filter = lambda x: {"radius": x, "range_filter": 1}

        mv_client.create_collection(
            collection_name=params.COLLECTION_NAME,
            schema=collection_schema,
            index_params=index_params,
        )
        mv_client.load_collection(collection_name=params.COLLECTION_NAME)
        log.debug('Vector store up')
        return mv_client

    def ensure_tables(self):
        """
        Makes sure the sqlite tables we need exist
        :return:
        """
        log.info("BeliefStore.ensure_tables")
        res = self.conn.execute("select name from sqlite_master")
        tables = [x[0] for x in res.fetchall()]
        for t in params.tables.keys():
            if t not in tables:
                self.conn.execute(params.tables[t])
                log.debug(f"Created table {t}")
        res = self.conn.execute("select * from sources where label='default'")
        # should live in the db and not be deleted
        if len(res.fetchall()) == 0:
            self.conn.execute(f"""insert into sources(url, label, description, credibility)\
            values('{params.d_src['url']}', '{params.d_src['label']}', \
             '{params.d_src['description']}', {params.d_src['credibility']})""",  )
            self.conn.commit()

    def exit(self, dump_tables=[], dump_vectors=False):
        """ Exit in an orderly way."""
        rv = ''
        log.info("BeliefStore.exit")
        beliefs.Belief.update_table(self)
        Support.update_table(self)
        for t in dump_tables:
            rv += f"----{t}----\n{self.dump_table(t)}\n"
        # same for bset
        self.mv_client.flush(collection_name=params.COLLECTION_NAME)
        #num_rows = self.mv_client.num_entities(collection_name=COLLECTION_NAME)
        #blog.debug(f"Num milvus rows: {num_rows}")
        #
        # print(self.mv_client.list_collections())
        if dump_vectors:
            rv += ('Dumping vector store')
            res = self.mv_client.query(collection_name=params.COLLECTION_NAME, filter='stmt_id >= 0')
            for row in res:
                rv += 'row: ' +  str(row['belief_id'])
        #
        self.mv_client.close()
        self.conn.close()
        log.info("bstore exited")
        return rv

    def get_embedding(self, text: str) -> list[float]:
        """
        returns the embedding fro the text as a list. Embedding model is set
        in the params file

        :param text: text to embed
        :return: list of floats or None
        """
        log.debug(f"BeliefStore.get_embedding: {text}")
        try:
            resp = ollama.embed(model=params.EMBEDDING_MODEL, input=text)
            embedding = resp.embeddings[0]
        except Exception as e:
            log.error(f"Cannot get embedding for {text}\n{str(e)}")
            embedding = None
            raise e
        return embedding

    def get_stmt_id(self,
                    txt):
        """
        gets tje statement id for the text

        :param txt: must match a statement
        :return: the id or exception
        """
        id_dist, is_new = self.get_similar_stmts(txt, wide_net=False,
                                         max_match=1)
        if len(id_dist) > 0:
            return id_dist[0][0]
        else:
            raise ValueError("Cannot get statement for " + txt)

    @retry(retry=retry_if_exception(Exception), stop=stop_after_attempt(params.VDB_RETRIES),
           after=after_log(log, logging.WARN))
    def get_similar_stmts(self,
                          txt: str,
                          wide_net: bool = False,
                          max_match: int=10,):
        """
        given a text return statements that are close to the text

        if the txt is new, add a stmt for it. this should be separated perhaps
        :param txt: new text
        :param max_dist: how far to accept
        :param max_match: how many to get
        :return: a list od [stmt_id, dist], isNew where the first stmt is the text
        """
        log.info("get_similar_stmts " + txt)
        id_dists = []
        is_new = False
        # assuming cosine similarity. generlaize later
        if wide_net:
            radius = params.COSINE_WN_RADIUS
            range = params.COSINE_WN_RANGE
        else:
            radius = params.COSINE_NN_RADIUS
            range = params.COSINE_NN_RANGE
        try:
            embedding = self.get_embedding(txt)
            # returns class whose first dimension is the number of vectors to query (nq),
            # and the second dimension is the number of limit (topk).
            # search params for similarity measure: between radius and range_filter
            #radius, range_filter = 0.3, 1
            id_dists = self.get_milvus_id_dist(embedding, radius, range, max_match)
            # this is list of [stmt_id, dist to the text]
        except Exception as e:
            log.error(f"Connot find siumilar to {txt}\n{str(e)}")
            raise e
        id_dists.sort(key=lambda x: x[1])
        the_stmt = None
        # get a stmt with the txt, either existing or a new one
        if len(id_dists) > 0:
            if id_dists[0][1] <= params.EPS_EMBED_IDENT:
                try:
                    # what if there are many of those? later
                    sql = f"select text from stmts where id = {id_dists[0][0]}"
                    res = self.conn.execute(sql)
                    stmt_text = res.fetchone()[0]
                except Exception as e:
                    log.error(f"Cannot get stmt for {id_dists[0][0]}\n{str(e)}")
                    raise e
                if Levenshtein.ratio(stmt_text, txt) >= params.LEVENSHTEIN_LB :
                    log.debug(f"have identical text {stmt_text} || {txt}")
                    the_stmt = id_dists[0][0]
        # get a new stmt for the txt
        if the_stmt is None:
            print('insertin ', txt)
            try:
                cur = self.conn.cursor()
                sql = f"insert into stmts(text) values(?)"
                cur.execute(sql, [txt])
                the_stmt = cur.lastrowid
                log.debug(f"Inserted statement {txt} with id {the_stmt}")
                cur.close()
                self.conn.commit()
                id_dists = [[the_stmt, 0]] + id_dists
                is_new = True
            except Exception as e:
                log.error(f"Cannot write new stmt {txt}\n{str(e)}")
                raise e
            try:
                res = self.mv_client.insert(collection_name=params.COLLECTION_NAME,
                                            data = {'stmt_id': the_stmt,
                                                    'embedding': embedding}
                                            )
                log.debug('milvus insertion ' + str(res))
            except Exception as e:
                log.error(e)
                raise e
            # add the distances computed to the distance table
            # recompute distances if we did narorw search above
            if radius != params.COSINE_WN_RADIUS or max_match <  params.SIM_NUM_MATCH:
                log.debug('Redoing vector search')
                radius = params.COSINE_WN_RADIUS
                range = params.COSINE_WN_RANGE
                id_dists = self.get_milvus_id_dist(embedding, radius, range, params.SIM_NUM_MATCH)
            values = []
            for sd in id_dists:
                log.debug(f"id-dists: {sd[0]}, {sd[1]}")
                ident = False
                if sd[0] > the_stmt:
                    values.append([the_stmt, sd[0], params.EMBEDDING_METRIC, sd[1]])
                elif sd[0] < the_stmt:
                    values.append([sd[0], the_stmt, params.EMBEDDING_METRIC, sd[1]])
                else:
                    ident = True
                if not ident:
                    log.debug(f'stmtdist: {values[-1][0]} {values[-1][1]} : {values[-1][3]}')
            if len(values) > 0:
                try:
                    ins = "insert into stmtdists(stmt_id_1, stmt_id_2, dtype, value) values(?, ?, ?, ?)"
                    res = self.conn.executemany(ins, values)
                    self.conn.commit()
                except Exception as e:
                    log.error(f"Cannot update stmtdists\n{str(e)}")
                    raise e
        # return the stmt that matches the txt exactly
        assert id_dists[0][1] <= params.EPS_EMBED_IDENT
        return id_dists, is_new

    def get_milvus_id_dist(self,
                           embedding,
                           radius,
                           range,
                           max_match):
        """
        return closest matches for the embedding

        :param embedding: to find closest matches
        :param radius: min similarity
        :param range: max similarity
        :param max_match: number to retuirn
        :return:
        """
        id_dists = []
        log.debug(f"get_milvus_id_dist radius: {radius}, range: {range}")
        res = self.mv_client.search(collection_name=params.COLLECTION_NAME,
                                    data=[embedding],
                                    search_params={
                                        'metric_type': params.EMBEDDING_METRIC,
                                        'params': {
                                            'radius': radius,
                                            'range_filter': range
                                        },
                                    },
                                    output_fields=['stmt_id'],
                                    limit=max_match,
                                    )
        # COSINE returns similarity. convert all to distances 0..1
        print('milvus output (similarities)\n', res)
        if len(res[0]) > 0:
            for x in res[0]:
                if 'id' in x:
                    dist = 1 - abs(x['distance']) if params.EMBEDDING_METRIC in ['COSINE', 'IP'] else x['distance']
                    id_dists.append([x['id'], dist])
                    log.debug(f"id_dist: {x['id']}, {dist}")
        else:
            log.debug('no results')
        return id_dists

    def get_similar_beliefs(self,
                            txt:str,
                            beliefset_id: int,
                            max_dist: float = 1.0,
                            max_match: int=10,
                            mult_match: bool = False):
        """
        get beliefs that are close to this text in the same beliefset

        TODO: get a list of similar stmts from get_similar_stmts to skip one join
            separate getting similar beleif/stmt from generating the stmtdists for new ones
        :param txt: text to get beleifs for
        :param beliefset_id: which set to look in
        :param max_dist: how far away?
        :param max_match: how many to retrieve?
        :param mult_match: do we always look for multiple matches even if there
            is an exact match?
        :return: list of [beleif-id, distance, text], the stmt dor this string
        """
        log.info(f"get_simiar_beleifs {txt}")
        id_dists, is_new = self.get_similar_stmts(txt, wide_net=True, max_match=max_match)
        bid_dist_txt = []
        the_stmt_id = id_dists[0][0]
        log.debug(f'best match {id_dists[0][0]}, {id_dists[0][1]}')
        log.debug(f"stmt id from get-similar-stmts: {the_stmt_id}")
        if id_dists[0][1] < params.EPS_EMBED_IDENT and not mult_match:
            # if the closest stmt/belief is almost identical, return just that one
            sql = """select b.id, 0.0, c.text
            from stmt2bel as a 
            join beliefs as b on a.belief_id=b.id
            join stmts as c on a.stmt_id=c.id
            where c.id=? and b.bset_id=?
            """
            try:
                res = self.conn.execute(sql, (the_stmt_id, beliefset_id))
                bid_dist_txt = res.fetchall()
            except Exception as e:
                log.error(f"Cannot get db for {sql}\n{str(e)}")
                raise e
        else:
            # otherwise return all those that are close enough
            sql = """select c.id, a.value, d.text 
            from stmtdists as a
            join stmt2bel as b on a.stmt_id_1 = b.stmt_id
            join beliefs as c on c.id = b.belief_id
            join stmts as d on d.id = a.stmt_id_1
            where a.stmt_id_2 = ? and c.bset_id = ? and a.value < ?;
            """
            try:
                res = self.conn.execute(sql, (the_stmt_id, beliefset_id, max_dist))
                bid_dist_txt = res.fetchall()
            except Exception as e:
                log.error(f"Cannot get db for {sql}\n{str(e)}")
                raise e
            # switch the ids
            sql = """select c.id, a.value, d.text 
            from stmtdists as a
            join stmt2bel as b on a.stmt_id_2 = b.stmt_id
            join beliefs as c on c.id = b.belief_id
            join stmts as d on d.id = a.stmt_id_2
            where a.stmt_id_1 = ? and c.bset_id = ? and a.value < ?;
            """
            try:
                res = self.conn.execute(sql, (the_stmt_id, beliefset_id, max_dist))
                bid_dist_txt.extend(res.fetchall())
            except Exception as e:
                log.error(f"Cannot get db for {sql}\n{str(e)}")
                raise e
        for m in bid_dist_txt:
            log.debug(f"match: {str(m)}")
        return bid_dist_txt, the_stmt_id, is_new

    def dump_vector_store(self):
        log.info('Dumping vector store')
        res = self.mv_client.query(collection_name=params.COLLECTION_NAME, filter='stmt_id >= 0')
        for row in res:
            print('row: ', row['stmt_id'])

    def dump_table(self, tname):
        res = self.conn.execute(f"select * from {tname}")
        lor = res.fetchall()
        rv = ''
        rv += f"-----------{tname}-----------------\n"
        for row in lor:
            rv += str(row) + '\n'
        return rv

    def get_belief_row_by_id(self, bid:int = None):
        """
        Given an id, it returns the belief from  db

        :param id:
        :return: the sql row
        """
        log.debug(f"BeliefStiore.get_belief_row_by_id {bid}")
        row = None
        if bid is not None:
            try:
                res = self.conn.execute("select * from beliefs where id=?", (bid, ))
                row = res.fetchone()
            except Exception as e:
                log.warning(f"Cannot get row {bid}\n{str(e)}")
                raise e
        if row is None:
            raise ValueError(f"Cannot find belief in db {bid}")
        return row

    def get_support_row_by_id(self, sid:int = None):
        """
        Given a support id, return the support object

        :param sid: support id
        :return: support object
        """
        log.debug(f"BeliefStiore.get_support_row_by_id {sid}")
        row = None
        if sid is not None:
            try:
                res = self.conn.execute("select * from supports where id=?", (sid, ))
                row = res.fetchall()[0]
            except Exception as e:
                log.warning(f"Cannot get support row {sid}\n{str(e)}")
                raise e
        if row is None:
            return None
        return row

    def list_bsets(self):
        rows = []
        sql = "select id, path, description from bsets"
        try:
            res = self.conn.execute(sql)
            rows = res.fetchall()
        except Exception as e:
            log.warning(f"Cannot get bset list \n {str(e)}")
        return rows



class BSContext:
    """
    context manager for the bstore to make sure we exit nicely
    TODO:
        add a __call__ method to set the location of the databases iso going
        with the default.
        https://stackoverflow.com/questions/68300246/passing-arguments-to-context-manager
    """
    bstore = None

    def __enter__(self):
        if BSContext.bstore is None:
            BSContext.bstore = BeliefStore()
        return BSContext.bstore

    def __exit__(self, type, value, traceback):
        if self.bstore is None:
            print('bstore cannot be None')
            raise ValueError('bstore is None')
        BSContext.bstore.exit()
        BSContext.bstore = None


