import argparse
import pathlib
import torch
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from transformer import get_transformer

FRAME_LEN=5
EXPRESSION_SPACE_DIM = 64

def process_x(betas_x, flatten=True):
    n_frames = len(betas_x)
    end_idxs = [n_frames-(FRAME_LEN-i-1) for i in range(FRAME_LEN)]
    end_idxs[-1] = None
    betas_x = [betas_x[i:end_idx, :EXPRESSION_SPACE_DIM] for i,end_idx in zip(range(FRAME_LEN), end_idxs)]
    result = torch.stack(betas_x, axis=1)
    if flatten:
        result = result.reshape((n_frames-FRAME_LEN+1), -1)
    return result

def process_y(betas_y, flatten=True):
    result = betas_y[(FRAME_LEN-1):, :EXPRESSION_SPACE_DIM]
    if not flatten:
        result = result[:,None,:]
    return result
    
def load_xy(source_list, target_list, flatten=True):
    for file in [*source_list, *target_list]:
        assert file.exists(), f"File {file} does not exist"
        assert file.is_file(), f"Path {file} is not a file"
    assert len(source_list) == len(target_list), "Source betas should correspond 1-1 to target betas, but got different lengths"
    source_stems = sorted([pathlib.Path(source).stem for source in source_list])
    target_stems = sorted([pathlib.Path(target).stem for target in target_list])
    assert source_stems == target_stems, "Source betas should correspond 1-1 to target betas, but got different file stems"
    assert len(set(source_stems)) == len(source_stems), "Duplicate file stems found!"
    assert len(set(target_stems)) == len(target_stems), "Duplicate file stems found!"
    x = [torch.load(source).detach() for source in tqdm(source_list)]
    y = [torch.load(target).detach() for target in tqdm(target_list)]
    x = [process_x(b, flatten=flatten) for b in x]
    y = [process_y(b, flatten=flatten) for b in y]
    x = torch.cat(x, axis=0)
    y = torch.cat(y, axis=0)
    return x, y

def load_x(source_list, flatten=True):
    for file in source_list:
        assert file.exists(), f"File {file} does not exist"
        assert file.is_file(), f"Path {file} is not a file"
    x = [torch.load(source).detach() for source in tqdm(source_list)]
    x = [process_x(b, flatten=flatten) for b in x]
    x = torch.cat(x, axis=0)
    return x

def get_automl(_):
    from autosklearn.regression import AutoSklearnRegressor
    import autosklearn
    return AutoSklearnRegressor(
        time_left_for_this_task=60*4,
        # time_left_for_this_task=60*60*4,
        # per_run_time_limit=60*6*4,
        memory_limit=6*1024,
        metric=autosklearn.metrics.mean_squared_error,
        n_jobs=5,
    )

def get_linear(_):
    from sklearn.linear_model import LinearRegression
    return LinearRegression()

zoo = {
    "automl": get_automl,
    "linear": get_linear,
    "transformer": get_transformer,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas-target", type=pathlib.Path, nargs="+", help="Used for scoring")
    parser.add_argument("--betas-source", type=pathlib.Path, nargs="+")
    parser.add_argument("--checkpoint", type=argparse.FileType("rb"), help="Path to a pickled model")
    parser.add_argument("--model", choices=zoo.keys(), default="automl")
    parser.add_argument("--output", type=argparse.FileType("wb"), default=None)
    args = parser.parse_args()
    flatten = (args.model != "transformer")
    if args.betas_target:
        betas_source, betas_target = load_xy(args.betas_source, args.betas_target, flatten=flatten)
    else:
        betas_source = load_x(args.betas_source, flatten=flatten)
        betas_target = None
    model = pickle.load(args.checkpoint)
    if betas_target is not None:
        score = model.score(betas_source, betas_target)
        print(f"Score: {score}")
    if args.output:
        pred = model.predict(betas_source)
        torch.save(pred, args.output)

    