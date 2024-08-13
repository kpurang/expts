!/usr/bin/env python
 coding: utf-8

"""
 News search and clustering

Input: keywords
Output: a set of articles relating to that keyword organized by event, event type and perspective.

Uses:
- gpt
- text-embedding-ada
- DBSCAN
- kMEans

 Main method: getNews(<Keywords>, <numberOfArticles>). 
 
 Given the keywords, it retrieves <numberOfArticles> articles from mrdiatack and organizes them into events and event types. For instance, Bart derailment is an event for which there can be multiple articles. If there are other transportation related articles the corresponding events can be in a transportation event type. A summary is generated for each event and event type.
 
 The method returns:
 1. a list of articles as obtained from Medistack with additional fields for events, event types and perspectives
 2. these articles arranged in a hierarchy with event types and events and including the summaries.
 
"""


import os
import http.client
import urllib.parse
from typing import Callable
import openai
from datetime import datetime
import time
import numpy
from scipy.spatial.distance import pdist, cdist
from sklearn.cluster import DBSCAN, KMeans
import logging

MODEL = 'gpt-4'
EMODEL = 'text-embedding-ada-002'

MEDIASTACK_API_KEY = os.environ['MEDIASTACK_API_KEY']
conn = http.client.HTTPConnection('api.mediastack.com')

EMBEDCHUNKSIZE = 10    # number of articles to embed in one chunk
CHATCHUNKSIZE = 10     # same for chat completion
DUPEDISTANCE = 0.01    # max distance between duplicate articles 
RELDISTANCE = 0.3      # max distance between keyword and article to count as relevant
DBSCAN_MINSAMPLES = 2  # min_samples for dbscan
DBSCAN_EPS = 0.1       # eps for dbscan
KMEANS_NCLUSTERS = 7   # num clusters for KMeans

PWD = os.getcwd()
SAVEALL = True         # save intermediate results

LOGFILE = os.path.join(PWD, "news.log")
logging.basicConfig(format='%(levelname)s: %(message)s',
                    filename=LOGFILE, 
                    level=logging.DEBUG)


# ## API queries
# 
# Generic methods to do retries and query mediastack, openai chat and embedding.


def retry(f: Callable = None, 
          argMap: dict = {},
          nRetries: int = 5,
          backoff: int = 2,):
    """
    Retry with exponential backoff
    """
    for atry in range(nRetries):
        try:
            rv = f(**argMap)
            return rv
        except Exception as e:
            if atry >= nRetries:
                raise e
            else:
                logging.warning(f"Try {atry}. exception: {str(e)}")
        waitTime = backoff * (2 ** atry)
        time.sleep(waitTime)

def queryMediastack(keywords:str='',
                   number:int=100) -> str:
    """
    Only the first page of english language articles which are ordered
    by publication date.
    """
    logging.debug('mediastack query')
    params = urllib.parse.urlencode({
        'access_key': MEDIASTACK_API_KEY,
        'keywords': keywords,
        'sort': 'published_desc',
        'limit': number,
        'languages': 'en',
        })
    conn.request('GET', '/v1/news?{}'.format(params))
    res = retry(conn.getresponse)
    # should reset connection if failure
    if res.status != 200:
        logging.warning(f"status: {res.status}: {res.reason}")
        raise ValueError(f"status {res.status}")
    data = res.read()
    return data.decode('utf-8')

def queryOpenai(query:list[dict] = [],
               instr:str = 'You are a helpful assistant.',
               temperature:float= 0.6,
               num_completions:int = 1,
               response_format:str = 'text',
               verbosity:int = 0) -> (str, list[str]):
    """
    Query openai chat completion
    """
    logging.debug('openai query')
    qObj = [{"role": "system", "content": instr}]
    qObj.extend(query)
    try:
        response = retry(openai.ChatCompletion.create,
                        {'model': MODEL, 'messages': qObj, 
                         'temperature': temperature, 'n': num_completions,
                          #'response_format': {'type': response_format},
                        }
                    )
    except Exception as e:
        logging.error('Repeated LLM failures\n' + str(e))
        return None, None
    answers = [x['message']['content'] for x in response['choices']]
    return response, answers


# # Openai chat prompts
# 
# openai chat completion is use for
# 1. labeling article by the event they are about, the event type and the perspective of the article.
# 2. finding labels for groups of events or event types
# 3. recursively summarizing groups of articles
# 



# Given a list of articles, use gpt4 to label them, adding the labels to the 
# article structure
# use the title and description for generating the labels
def labelChunk(articles: list[dict],
                 events:list[str] = [],
                 eventTypes:list[str] = [],
                 perspectives:list[str] = [],) -> list[dict]:
    logging.debug('labeling a chunk')
    artAndDescr = [{'id': i, 'title': x['title'], 'description': x['description']} for i, x in enumerate(articles)]
    sysInstr = "You are an experienced news editor. "
    queryStr = f"""Given the news article titles and descriptions in the json list below, \
    for each article, find the event the article is about, the event type and the perspective \
    of the author of the article. Use the events, event types and perspectives listed 
    below, or if they are not appropriate, add new ones.
    Your only output should be a valid json list of dicts with the following keys: 'id', 'title', 'event', \
    'event_type', 'perspective' corresponding to each of the articles in the inputs. \
    Make sure the json you generate is valid. The json output must be constructed with \
    double-quotes. Double quotes within strings must be escaped with backslash, \
    single quotes within strings will not be escaped.
    
    articles: {artAndDescr}

    events: {events}

    event_types: {eventTypes}

    perspectives: {perspectives}

    """
    response, answers = retry(queryOpenai,
                              {'query': [{'role': 'user', 'content': queryStr}],
                               'response_format': 'json_object'}
                             )
    return response, answers

def labelClusters(labels: list[str]) -> str:
    """
    Given a list of labels that are clustered together, return a label that
    represents the cluster. 
    This will be used to map the articles.
    """
    logging.debug('labeling clusters')
    sysTnstr = "You are an experienced news editor."
    queryStr = f"""Given the following list of labels, find a label that \
    represents all of them. The label you generate should be similar in \
    length to the labels given. Only return the new label.

    Example:
    Labels: dog bed, cat tree, dog toy, rabbit hutch, fish tank
    New label: pet supplies

    Labels: {', '.join(labels)}
    New label:
    """
    response, answers = retry(queryOpenai,
                              {'query': [{'role': 'user', 'content': queryStr}]}
                               )
    return response, answers

def summarizeNews(texts: list[str]) -> str:
    """
    Given a list of strings: title + description, summarize them
    """
    logging.debug('summarizing news')
    sysInstr = "You are an experienced news editor."
    queryStr = f"""Summarize the following list of descriptions of news articles \
    concisely to capture the more interesting parts of the news.

    Descriptions: {texts}

    Summary:
    """
    response, answers = retry(queryOpenai,
                              {'query': [{'role': 'user', 'content': queryStr}]}
                               )
    return response, answers

    


# ## Cleaning articles
# 
# Articles are deduped and those not relevant to the keywords are removed. 
# 
# Relevance is computed from cosine distance between the keyword and the article and duplicates are computed using cosine distance between pairs of articles. Embeddings are from openai and scipy methods cdist and pdist are used for the computations.
# 


def embedChunk(list2embed:list[str]) -> list[list[int]]:
    logging.debug('embedding chunk')
    try:
        embeds = retry(openai.Embedding.create,
                       {'input': list2embed,
                        'model': EMODEL})
    except Exception as e:
        logging.error(f"Cannot embed: {e}")
        return []
    embeddings = [x['embedding'] for x in embeds['data']]
    return embeddings

def cleanArticles(articles:list[dict] = [], 
                 keywords:str = '') -> list[dict]:
    """
    remove irrelevant and duplicate articles
    we are using only the ttile and description, so things might be a bit off
    chunk articles in groups 
    """
    logging.debug('cleaning articles')
    emb = []
    for i in range(0, len(articles), EMBEDCHUNKSIZE):
        artStrs = [f"{x['title']} {x['description']}" for x in articles[i: i+EMBEDCHUNKSIZE]]
        someE = embedChunk(artStrs)
        assert len(someE) == len(articles[i: i+EMBEDCHUNKSIZE])
        emb.extend(someE)
    artEmbeddings = np.array(emb)
    # TODO: handle negative keywords inversely. for now just remove them
    # TODO: also have an embedding for each individual tern in the keywords
    kwdList = [x.strip() for x in keywords.split(',')]
    cleanKwds = [x for x in kwdList if not x.startswith('-')]
    cKwdStr = ','.join(cleanKwds)
    try:
        kwdE = retry(openai.Embedding.create,
                       {'input': [cKwdStr],
                        'model': EMODEL})
    except Exception as e:
        logging.error(f"Cannot embed: {e}")
        raise e
    kwdEmbeddings = np.array([kwdE['data'][0]['embedding']])
    kwdDistances = cdist(artEmbeddings, kwdEmbeddings, metric='cosine')
    irrelevants = []
    for i in range(len(articles)):
        if kwdDistances[i][0] >= RELDISTANCE:
            irrelevants.append(i)
    for ra in zip([x['title'] for x in articles], kwdDistances):
        logging.debug(f"{ra[1][0]} : {ra[0]}")
    logging.info(f"Irrelevant: {chr(10).join([articles[i]['title'] for i in irrelevants])}")
    # The metric dist(u=X[i], v=X[j]) is computed and stored in entry 
    #  m * i + j - ((i + 2) * (i + 1)) // 2.
    articleDistances = pdist(artEmbeddings, metric='cosine')
    dupes = set()
    for i in range(len(articles)):
        for j in range(len(articles)):
            if i >= j: continue
            idx = len(articles) * i + j - ((i+2)*(i+1)) // 2
            if articleDistances[idx] <= DUPEDISTANCE:
                logging.info(f"dupes: {articleDistances[idx]}\n{articles[i]['title']}\n{articles[j]['title']}")
                dupes.add(j)
    goodArticles = articles.copy()
    allToDel = sorted((irrelevants + list(dupes)), reverse=True)
    for i in allToDel:
        del goodArticles[i]
    return goodArticles


# ## Labeling articles
# 
# Articles are zero-shot labeled with the event they are about the type of event and the perspective of the author. This is done in chunks of articles with the LLM asked to 
# reuse categories for later chunks.


def labelArticles(articles: list[dict]) -> list[dict]:
    logging.debug('labeling articles')
    events = {}
    eventTypes = {}
    perspectives = {}
    for i in range(0, len(articles), CHATCHUNKSIZE):
        articlesToLabel = articles[i : i+CHATCHUNKSIZE]
        _, answers = labelChunk(articlesToLabel,
                                list(events.keys()),
                                list(eventTypes.keys()),
                                list(perspectives.keys()))
        try:
            labeled = json.loads(answers[0])
        except Exception as e:
            logging.error(f"Cannot read json from openai\n{str(e)}")
            for k in range(i, i+len(articlesToLabel)):
                articles[k]['event'] = 'None'
                articles[k]['eventType'] = 'None'
                articles[k]['perspective'] = 'None'
            continue
        assert len(articlesToLabel) == len(labeled)
        for j, x in enumerate(labeled):
            if x['event'] in events: events[x['event']] += 1
            else: events[x['event']] = 1
            if x['event_type'] in eventTypes: eventTypes[x['event_type']] += 1
            else: eventTypes[x['event_type']] = 1
            if x['perspective'] in perspectives: perspectives[x['perspective']] += 1
            else: perspectives[x['perspective']] = 1
            articles[i+j]['event'] = x['event']
            articles[i+j]['eventType'] = x['event_type']
            articles[i+j]['perspective'] = x['perspective']
    logging.info(f"events: {','.join(events)}")
    logging.info(f"eventTypes: {','.join(eventTypes)}")
    logging.info(f"perspectives: {','.join(perspectives)}")
    return articles, events, eventTypes, perspectives

 


# ## Clustering
# 
# The LLM generally generates many events and eventypes. These are clustered to make it easier to understand. The clustering is based on embeddings and two algorithms are used: 1. DBSCAN for events - the size of the neighborhood is specified so we get closely related events.
# 2, KMeans for event types - the number of event types desired is a parameter of KMeans, so we get a small number of top level categories.
# 



# given a list of strings, uses dbscan to cluster them using their embeddings
def clusterStringsDBSCAN(strList: list[str]=[],
                  minSamples:int = DBSCAN_MINSAMPLES,
                  eps:float = DBSCAN_EPS) -> dict[int, list[str]]:
    logging.debug('DBSCAN')
    embList = embedChunk(strList)
    embeddings = np.array(embList)
    dbscan = DBSCAN(eps = eps, min_samples = minSamples, metric='cosine')
    clusters = dbscan.fit(embeddings)
    logging.info(f"DBScan labels: {str(clusters.labels_)}")
    groups = {}
    for el in zip(strList, clusters.labels_):
        if el[1] in groups:
            groups[el[1]].append(el[0])
        else:
            groups[el[1]] = [el[0]]
    return groups

# here we use kemans
# TODO: merge this with the above 
def clusterStringsKMeans(strList: list[str] = [],
                         n_clusters:int = KMEANS_NCLUSTERS,
                        ) -> dict[int, list[str]]:
    logging.debug('KMeans')
    embList = embedChunk(strList)
    embeddings = np.array(embList)
    kmeans = KMeans(n_clusters = n_clusters, init='random', n_init='auto')
    clusters = kmeans.fit(embeddings)
    logging.info(f"KMeans labels: {str(clusters.labels_)}")
    groups = {}
    for el in zip(strList, clusters.labels_):
        if el[1] in groups:
            groups[el[1]].append(el[0])
        else:
            groups[el[1]] = [el[0]]
    return groups
    


# ## Re-labeling
# 
# Once the events and event types have been clustered, the clusters need a new label. This is one using gpt-4.



def updateLabels(labeledArticles: list[dict],
                 eventList: list[str],
                 eventTypesList: list[str],
                ) -> list[dict]:
    logging.debug('Updating labels')
    """
    Given the labeled articles and the list of events and eventTypes, cluster
    these and relabel the articles.
    The aim is to group articles in sensible clusters for the user
    """
    eventTypeClusters = clusterStringsKMeans(eventTypesList)
    eventTypeMap = {}
    newEventTypes = []
    for l in eventTypeClusters.keys():
        if l == -1: continue
        elist = eventTypeClusters[l]
        _, ll = labelClusters(elist)
        newEventTypes.append(ll[0])
        for e in elist:
            eventTypeMap[e] = ll[0]
    logging.info(f"EventTypeMap:\n'{str(eventTypeMap)}")
    # relabel eventTypes
    for a in labeledArticles:
        if a['eventType'] in eventTypeMap: 
            a['eventType'] = eventTypeMap[a['eventType']]
    # cluster and label events for each eventType
    for et in newEventTypes:
        articles = [x for x in labeledArticles if x['eventType'] == et]
        eventList = [x['event'] for x in articles]
        logging.info(f"TYpe {et}, events: {eventList} num articles {len(articles)}")
        # if len(eventList) < 3: continue
        eventClusters = clusterStringsDBSCAN(eventList)
        eventMap = {}   # only for clustered events
        # cluster the labels
        for l in eventClusters.keys():
            if l == -1: continue
            elist = eventClusters[l]
            _, ll = labelClusters(elist)
            for e in elist:
                eventMap[e] = ll[0]
        logging.info(f"eventMap: \n'{str(eventMap)}")
        eventCount = {}
        for a in articles:
            if a['event'] in eventMap: 
                a['event'] = eventMap[a['event']]
            if a['event'] not in eventCount: eventCount[a['event']] = 0
            eventCount[a['event']] += 1
        for a in articles:
            if eventCount[a['event']] == 1:
                a['event'] = 'Misc.'
    return labeledArticles
            


# ## Summarizing the groups
# 
# For each event, the articles for that event are summarized into an event summary. And all events in an event type are summarized to give the event type summary.
# 


def genSummaries(labeledArticles: list[dict]) -> dict:
    """
    given the updated labeled articles, summarize events and event types and 
    return a map eventType -> summary. event -> summary, [articles]
    """
    logging.debug('generating summaries')
    news = {}
    for a in labeledArticles:
        if a['eventType'] not in news:
            news[a['eventType']] = {}
        if a['event'] not in news[a['eventType']]:
            news[a['eventType']][a['event']] = {'articles':[]}
        news[a['eventType']][a['event']]['articles'].append(a)
    for et in news.keys():
        eventSummaries = []
        for e in news[et].keys():
            descrs = [x['description'] for x in news[et][e]['articles']]
            _, ans = summarizeNews(descrs)
            summary = ans[0]
            logging.info(f"{e} summary: {summary}")
            news[et][e]['summary'] = summary
            eventSummaries.append(summary)
        if len(eventSummaries) > 1:
            _, ans = summarizeNews(eventSummaries)
        else: ans = eventSummaries
        news[et]['summary'] = ans[0]
        logging.info(f"{e} summary: {ans[0]}")
    return news


# ## Main method
# 
# getNews is the main method with parameters: keywords and number of articles to retrieve.


def getNews(keyword:str = 'San Francisco',
           numArticles:int = 100):
    startTime = time.time()
    rawData = json.loads(queryMediastack(keyword, number=numArticles))
    if SAVEALL:
        with open(os.path.join(PWD, 
                               f"{keyword.replace(' ', '_')}_raw.json"), 'w') as w:
            json.dump(rawData, w, indent=4) 
    goodArticles = cleanArticles(rawData['data'], keyword)
    print(f"Num good articles: {len(goodArticles)}")
    if SAVEALL:
        with open(os.path.join(PWD,
                               f"{keyword.replace(' ', '_')}_good.json"), 'w') as w:
            json.dump(goodArticles, w, indent=4) 
    labeledArticles, events, eventTypes, perspectives = labelArticles(goodArticles)
    print('Labeled articles')
    if SAVEALL:
        with open(os.path.join(PWD,
                               f"{keyword.replace(' ', '_')}_labeled.json"), 'w') as w:
            json.dump(labeledArticles, w, indent=4) 
    labeledArticles = updateLabels(labeledArticles, list(events.keys()), list(eventTypes.keys()))
    print('Updated labels')
    if SAVEALL:
        with open(os.path.join(PWD,
                               f"{keyword.replace(' ', '_')}_updlabels.json"), 'w') as w:
            json.dump(labeledArticles, w, indent=4) 
    summaries = genSummaries(labeledArticles)
    print('Generated summaries')
    if SAVEALL:
        with open(os.path.join(PWD,
                               f"{keyword.replace(' ', '_')}_summaries.json"), 'w') as w:
            json.dump(summaries, w, indent=4) 
    for et in summaries.keys():
        print(f"Event type: {et}\n\tSummary: {summaries[et]['summary']}")
        for e in summaries[et].keys():
            if e == 'summary': continue
            print(f"\tEvent: {e}\n\tSummary: {summaries[et][e]['summary']}")
            print("\tArticles:")
            for a in summaries[et][e]['articles']:
                print(f"\t\t{a['title']}")

        print('\n')
    print(f"\nRun time= {(time.time() - startTime):.2f}s")
    return summaries, labeledArticles


# getNews('city')

