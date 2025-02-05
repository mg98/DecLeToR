print("Importing modules...")
import pickle
import numpy as np
from argparse import ArgumentParser
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
import random
from baselines.panache import panache_rank
from baselines.dinx import dinx_rank, dinx_rank_by_seeders
from baselines.maay import maay_rank_numpy as maay_rank
from baselines.grank import grank_fast as grank
from baselines.random import random_rank
from baselines.tribler import tribler_rank
from baselines.ltr import ltr_rank
from common import UserActivity, mean_ndcg, timing
print("Done importing modules")

np.random.seed(42)

K_RANGE = [5, 10, 30, None]
NUM_CORES = cpu_count()

# generate a range of indices with increasing steps
def gen_fast_range(end):
    total_range = [0]
    i = 1
    while i < end:
        total_range.append(i)
        i = int(i * 1.05) + 1
    total_range.append(end)
    return total_range

def chronological_eval(user_activities: list[UserActivity], rank_func) -> dict[int, dict[int, float]]:
    TEST_DATA_SIZE = 100
    if rank_func.__name__ == "ltr_rank":
        NUM_WORKERS = min(NUM_CORES, 4)
        total_range = gen_fast_range(len(user_activities) - TEST_DATA_SIZE)
    else:
        NUM_WORKERS = min(NUM_CORES, 8)
        total_range = range(0, len(user_activities) - TEST_DATA_SIZE)

    def process_index(i):
        try:
            # Use array indexing instead of slicing to avoid copying
            context = user_activities[:i]
            # Create indices for random selection instead of copying data
            test_indices = random.sample(range(i, len(user_activities)), TEST_DATA_SIZE)
            test_data = [user_activities[idx] for idx in test_indices]
            
            ranked_user_activities = rank_func(context, test_data)
            ndcgs = {k: mean_ndcg(ranked_user_activities, k) for k in K_RANGE}
            return i, ndcgs
        except Exception as e:
            print(f"Error processing index {i}: {str(e)}")
            return None

    results = Parallel(n_jobs=NUM_WORKERS, batch_size=4)(
        delayed(process_index)(i) for i in tqdm(total_range, desc="Processing indices")
    )
    
    # Filter out None results and sort by context size
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x[0])
    
    context_to_ndcgs = {i: ndcgs for i, ndcgs in results}

    return context_to_ndcgs

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-n', type=int, help='Number of user activities to use')
    parser.add_argument('--ltr', action='store_true', help='Only run LTR algorithm')
    args = parser.parse_args()

    print(f"Number of CPU cores available: {NUM_CORES}")

    print("Loading user activities...")
    with open('user_activities.pkl', 'rb') as f:
        user_activities = pickle.load(f)
    np.random.shuffle(user_activities)
    if args.n:
        user_activities = user_activities[:args.n]
    print(f"Loaded {len(user_activities)} user activities.")

    ranking_algos = {
        "tribler": tribler_rank,
        "random": random_rank,
        "panache": panache_rank,
        "dinx": dinx_rank,
        "dinx_s": dinx_rank_by_seeders,
        "maay": maay_rank,
        "grank": grank
    }
    if args.ltr:
        ranking_algos = {"ltr": ltr_rank}

    all_ndcgs = {}

    for algo_name, ranking_algo in ranking_algos.items():
        print(f"============{algo_name}=============")
        
        with timing() as t:        
            ndcgs = chronological_eval(user_activities, ranking_algo)
            all_ndcgs[algo_name] = ndcgs
    
    if args.ltr:
        with open(f'results/context_ndcgs_ltr.pkl', 'wb') as f:
            pickle.dump(all_ndcgs, f)
    else:
        with open(f'results/context_ndcgs.pkl', 'wb') as f:
            pickle.dump(all_ndcgs, f)
    print('Success!')
