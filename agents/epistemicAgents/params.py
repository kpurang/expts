
# parameters

BASEDIR = '/Users/kp/projects/python/projects/agents/data'

## FILES
LOGFILE = '/Users/kp/projects/python/projects/agents/log/beliefs.log'
llm_log_fname = '/Users/kp/projects/python/projects/agents/logs/llm.log'
just_log_fname = '/Users/kp/projects/python/projects/agents/logs/just.log'

milvusLoc = '/Users/kp/projects/python/projects/agents/data/dbs/milvus_0529.db'
sqliteLoc = '/Users/kp/projects/python/projects/agents/data/dbs/sqlite_0529.db'
LLM_PROMPT_DIR = '/Users/kp/projects/python/projects/agents/epistemicAgents/prompts/tasks'

## URLS
ollama_generate_host = "http://localhost:11434/api/generate"
ollama_embedding_host = "http://localhost:11434/api/embed"


# milvus
COLLECTION_NAME = 'Statements'
EMBEDDING_METRIC = 'COSINE'     # DO NOT CHANGE FOR NOW
EPS_EMBED_IDENT = 1e-2
COSINE_WN_RADIUS = 0.25
COSINE_WN_RANGE = 1.2	# similarity can be > 1. maybbe better to round
COSINE_NN_RADIUS = 0.8
COSINE_NN_RANGE = 1.2
SIM_NUM_MATCH = 256     # max number of similar embeddings to find
VDB_RETRIES = 2

# sqlite
# add indices keys etc later
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
                        belief_id integer not null,
                        score float)""",
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
LLM_RETRIES = 2

# distances
LEVENSHTEIN_LB = 0.8
SB_THRESHOLD = 0.25     # max distance for similar beliefs
RB_THRESHOLD = 0.5      # max distance for related beliefs
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
# organize that better later
credibilities = {'axiom': 1.0,
                 'query': 0.0,
                 'llm': 0.6,
                 }


# reasoning parameters
BS_DEPTH = 5    # max depth for backward search
PROP_MIN_CHANGE = 0.1   # minimum change to propagate confidence changes

# num extra sources to add to class
SOURCE_LEN_ADD = 10

# for nl_utils
p_prop = 0.75  # max proportion of a line in parens for the paren contents to be deleted
max_dist = 0.2
justification_prefixes = ['fact', 'assumption', 'consequence', 'conclusion', 'inference']

min_inference_likslihood = 0.6

max_graph_label_len = 25
max_derivation_depth = 10
# https://graphviz.org/doc/info/colors.html
support2color = {'from_source': 'lightyellow',
                 'from_reasoning': 'gold',
                 'from_merge': 'gray95',
                 'from_axiom': 'cornsilk',
                 'from_query': 'palegreen',
                 'from_llm' : 'skyblue1'
}
