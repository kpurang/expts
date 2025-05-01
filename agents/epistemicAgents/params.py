
# parameters

LOGFILE = '/Users/kp/projects/python/projects/llm/beliefs/beliefMaintenance/log/beliefs.log'

# milvus
COLLECTION_NAME = 'Statements'
milvusLoc = '/tmp/kp/milvus_0.db'
EMBEDDING_METRIC = 'COSINE'     # DO NOT CHANGE FOR NOW
EPS_EMBED_IDENT = 1e-2
COSINE_WN_RADIUS = 0.25
COSINE_WN_RANGE = 1
COSINE_NN_RADIUS = 0.8
COSINE_NN_RANGE = 1
SIM_NUM_MATCH = 256     # max number of similar embeddings to find

# sqlite
# add indices keys etc later
sqliteLoc = '/tmp/kp/sqlite_0.db'
tables = {'stmts': """create table stmts(id integer primary key not null,
                    text varchar(512),
                    source_id integer)""",
          'stmtdists': """create table stmtdists(id integer primary key not null,
                          stmt_id_1 integer not null,
                          stmt_id_2 integer not null,
                          dtype integer default 0,
                          value float not null)""",
          'beliefs': """create table beliefs(id integer primary key not null, 
                          text_rep varchar(512),
                          bset_id integer,
                          support_id integer
                          )""",
          'stmt2bel': """create table stmt2bel(id integer primary key not null,
                        stmt_id integer not null,
                        belief_id integer not null)""",
          'bsets': """create table bsets(id integer primary key not null,
                          path varchar(256),
                          description varchar(256))""",
          'supports': """create table supports(id integer primary key not null,
                            belief_id id integer,
                            confidence float,
                            method varchar(128),
                            info varchar(256),
                            supported_by varchar(128),
                            supports varchar(128))""" ,
          'sources': """create table sources(id integer primary key not null,
                        source_type integer,
                        url varchar(128), 
                        label varchar(32),
                        description varchar(128),
                        credibility float)
          """,
          }

# default source
d_src = {'url': '', 'label': 'default', 'description': '', 'credibility': 0.7}

# ollama
EMBEDDING_MODEL = 'all-minilm'
EMBEDDING_DIM = 384
CHAT_MODEL = 'llama3.2'
SEMANTIC_MODELS = ['mistral', 'phi3', 'llama3.2']
LLM_PROMPT_DIR = '/Users/kp/projects/python/projects/llm/beliefs/beliefMaintenance/v0.3/prompts'
LLM_RETRIES = 5

# distances
LEVENSHTEIN_LB = 0.8
SB_THRESHOLD = 0.25     # max distance for similar beliefs
RB_THRESHOLD = 0.75     # max distance for related beliefs
# min abs llm similarity score to consider 2 sentences to mean the same thing
LLM_SIM_THRESHOLD = 0.7

# credibility of differnet sources. should be eventually part of beliefs and
# derived in some way
# these are 'label' in SInfo.info
source_credibility = {'s0': 0.7,
                      'bob': -0.8,
                      's2': 0.9,
                      'default': 0.7,
                      'axiom': 0.95,
                      'query': 0.0,
                      }

# reasoning parameters
BS_DEPTH = 5    # max depth for backward search
PROP_MIN_CHANGE = 0.1   # minimum change to propagate confidence changes

# num extra sources to add to class
SOURCE_LEN_ADD = 10


