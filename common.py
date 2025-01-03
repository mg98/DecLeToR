import sqlite3
import os
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd
import functools

np.random.seed(42)

def ranking_func(_func=None, *, shuffle=True):
    def _decorate(func):
        @functools.wraps(func)
        def wrapper(arg1, arg2=None):
            clicklogs = deepcopy(arg1)
            activities = deepcopy(arg2) if arg2 is not None else deepcopy(arg1)

            if shuffle:
                for ua in clicklogs:
                    np.random.shuffle(ua.results)
                
                for ua in activities:
                    np.random.shuffle(ua.results)

            return func(clicklogs, activities)
        return wrapper
    
    if _func is not None and callable(_func):
        return _decorate(_func)
    
    return _decorate

class TorrentInfo:
    title: str
    tags: list[str]
    timestamp: float
    size: int

    def __init__(self, **kwargs):
        self.title = kwargs.get('title', '')
        self.tags = kwargs.get('tags', [])
        self.timestamp = kwargs.get('timestamp', 0)
        self.size = kwargs.get('size', 0)

    def __getstate__(self):
        return {
            'title': self.title,
            'tags': self.tags,
            'timestamp': self.timestamp,
            'size': self.size
        }

    def __setstate__(self, state):
        self.title = state['title']
        self.tags = state['tags'] 
        self.timestamp = state['timestamp']
        self.size = state['size']

class UserActivityTorrent:
    infohash: str
    seeders: int
    leechers: int
    torrent_info: TorrentInfo

    def __init__(self, data):
        self.infohash = data['infohash']
        self.seeders = data['seeders'] 
        self.leechers = data['leechers']
        self.torrent_info = None

    def __str__(self):
        return f"Infohash: {self.infohash}, Seeders: {self.seeders}, Leechers: {self.leechers}, Torrent Info: {self.torrent_info}"
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self.torrent_info, TorrentInfo):
            state['torrent_info'] = {
                'title': self.torrent_info.title,
                'tags': self.torrent_info.tags,
                'timestamp': self.torrent_info.timestamp,
                'size': self.torrent_info.size,
            }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if isinstance(self.torrent_info, dict):
            self.torrent_info = TorrentInfo(**self.torrent_info)

class UserActivity:
    issuer: str
    query: str
    timestamp: int
    results: list[UserActivityTorrent]
    chosen_result: UserActivityTorrent

    def __init__(self, data: dict):
        self.query = data['query']
        self.timestamp = int(data['timestamp'] / 1000)
        self.results = []
        for result in data['results']:
            torrent = UserActivityTorrent(result)
            self.results.append(torrent)
        self.chosen_result = self.results[data['chosen_index']]

    def __repr__(self):
        return (f"UserActivity(issuer={self.issuer}, query={self.query}, "
                f"timestamp={self.timestamp}, chosen_result={self.chosen_result}, results={self.results})")

    def __getstate__(self):
        state = self.__dict__.copy()
        state['results'] = [result.__getstate__() for result in self.results]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Check if results contains UserActivityTorrent objects or dicts
        if state['results'] and not isinstance(state['results'][0], UserActivityTorrent):
            self.results = [UserActivityTorrent(result) for result in state['results']]
        # Update chosen_result to point to the correct object in results
        if self.chosen_result:
            chosen_hash = self.chosen_result.infohash
            self.chosen_result = next(r for r in self.results if r.infohash == chosen_hash)


def fetch_torrent_infos(user_activities: list[UserActivity]):
    """Fetch torrent info for a list of UserActivityTorrent objects using batched SQL queries"""
    all_torrents = [t for ua in user_activities for t in ua.results]
    infohashes = list(set(t.infohash for t in all_torrents))
    
    BATCH_SIZE = 50000
    torrent_info_map = {}
    
    conn = sqlite3.connect(os.path.expanduser('./metadata.db'))
    cursor = conn.cursor()

    for i in range(0, len(infohashes), BATCH_SIZE):
        batch = infohashes[i:i + BATCH_SIZE]
        placeholders = ','.join(['?' for _ in batch])
        
        cursor.execute(f"""
            SELECT infohash_hex, title, tags, timestamp/1000 as timestamp, size 
            FROM ChannelNode
            WHERE infohash_hex IN ({placeholders})
            """, batch)
        
        results = cursor.fetchall()
        
        for result in results:
            info = TorrentInfo()
            info.title = result[1]
            info.tags = result[2].split(',') if result[2] else []
            info.timestamp = result[3]
            info.size = result[4]
            torrent_info_map[result[0]] = info
    
    conn.close()
    
    # Update torrents in original user_activities
    found = 0
    not_found = 0
    for ua in user_activities:
        for torrent in ua.results:
            if torrent.infohash in torrent_info_map:
                torrent.torrent_info = torrent_info_map[torrent.infohash]
                found += 1

                if ua.chosen_result.infohash == torrent.infohash:
                    ua.chosen_result = torrent
            else:
                not_found += 1
    
    print(f'Found {found} torrents, skipped {not_found}')

    

class TFIDF:
    def __init__(self, corpus: Dict[str, str]):
        self.documents = list(corpus.values())
        self.doc_ids = list(corpus.keys())
        self.vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        # Precompute term counts for each document
        self.term_counts = [doc.split() for doc in self.documents]
        self.total_terms = [len(doc) for doc in self.term_counts]

    def get_tf_idf(self, doc_id: str, term: str) -> dict[str, float]:
        try:
            word_idx = list(self.feature_names).index(term)
        except ValueError:
            return { "tf": 0, "tf_idf": 0, "idf": 0 }
    
        doc_idx = self.doc_ids.index(doc_id)
        tf_idf = self.tfidf_matrix[doc_idx, word_idx]
        idf = self.vectorizer.idf_[word_idx]
        tf = tf_idf / idf if idf != 0 else 0
        
        return { "tf": tf, "tf_idf": tf_idf, "idf": idf }
    
@dataclass
class QueryDocumentRelationVector:
    tf_min: float = 0.0
    tf_max: float = 0.0
    tf_mean: float = 0.0
    tf_sum: float = 0.0
    tf_variance: float = 0.0
    idf_min: float = 0.0
    idf_max: float = 0.0
    idf_mean: float = 0.0
    idf_sum: float = 0.0
    idf_variance: float = 0.0
    tf_idf_min: float = 0.0
    tf_idf_max: float = 0.0
    tf_idf_mean: float = 0.0
    tf_idf_sum: float = 0.0
    tf_idf_variance: float = 0.0
    bm25: float = 0.0
    seeders: int = 0
    leechers: int = 0
    age: float = 0.0
    query_hit_count: int = 0

    def __str__(self):
        fields = [self.seeders, self.leechers, self.age, self.bm25, self.query_hit_count,
                 self.tf_min, self.tf_max, self.tf_mean, self.tf_sum,
                 self.idf_min, self.idf_max, self.idf_mean, self.idf_sum,
                 self.tf_idf_min, self.tf_idf_max, self.tf_idf_mean, self.tf_idf_sum,
                 self.tf_variance, self.idf_variance, self.tf_idf_variance]
        return ' '.join(f'{i}:{val}' for i, val in enumerate(fields))

class ClickThroughRecord:
    rel: float
    qid: int
    qdr: QueryDocumentRelationVector

    def __init__(self, rel=0.0, qid=0, qdr=None): 
        self.rel = rel
        self.qid = qid
        self.qdr = qdr

    def to_dict(self):
        return {
            'rel': self.rel,
            'qid': self.qid,
            'qdr': self.qdr
        }

    def __str__(self):
        return f'{self.rel} qid:{self.qid} {self.qdr}'
    
def split_dataset_by_qids(records, train_ratio=0.8, val_ratio=0.1):
    """
    Split records into train/validation/test sets based on query IDs.
    
    Args:
        records: list containing the records
        train_ratio: Proportion of data for training (default 0.8)
        val_ratio: Proportion of data for validation (default 0.1)
        
    Returns:
        tuple of (train_records, val_records, test_records) as lists of ClickThroughRecord objects
    """
    records_df = pd.DataFrame([record.to_dict() for record in records])
    qids = records_df['qid'].unique()
    # np.random.shuffle(qids) # For some reason, shuffling the qids leads to worse results
    
    # Calculate split sizes
    n_qids = len(qids)
    train_size = int(train_ratio * n_qids)
    val_size = int(val_ratio * n_qids)
    
    # Split qids into train/val/test
    train_qids = qids[:train_size]
    val_qids = qids[train_size:train_size+val_size]
    test_qids = qids[train_size+val_size:]
    
    # Filter records by qid
    train_records_df = records_df[records_df['qid'].isin(train_qids)]
    val_records_df = records_df[records_df['qid'].isin(val_qids)]
    test_records_df = records_df[records_df['qid'].isin(test_qids)]
    
    # Convert to ClickThroughRecord objects
    train_records = [ClickThroughRecord(**record) for _, record in train_records_df.iterrows()]
    val_records = [ClickThroughRecord(**record) for _, record in val_records_df.iterrows()]
    test_records = [ClickThroughRecord(**record) for _, record in test_records_df.iterrows()]
    
    return train_records, val_records, test_records