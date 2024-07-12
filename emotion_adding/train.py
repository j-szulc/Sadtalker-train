from inference import *
from automl import *
from transformer import *
from linear import *
from autosklearn.regression import AutoSklearnRegressor
import dill

zoo = {
    "automl": get_automl,
    "linear": get_linear,
    "transformer": get_transformer
}

def train(model, train_betas_source, train_betas_target, test_betas_source, test_betas_target, output=None):
    model.fit(train_betas_source, train_betas_target)
    if output is not None:
        output = pathlib.Path(output).with_suffix(".pkl")
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output), "wb") as f:
            dill.dump(model, f)
    train_score = model.score(train_betas_source, train_betas_target)
    print(f"Training score: {train_score}")
    if test_betas_source is not None:
        val_score = model.score(test_betas_source, test_betas_target)
        print(f"Validation score: {val_score}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas-target", type=pathlib.Path, nargs="+", help='"Ground-truth" betas')
    parser.add_argument("--betas-source", type=pathlib.Path, nargs="+", help="Betas from the original model")
    parser.add_argument("--model", choices=zoo.keys())
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model_name = args.model
    model = zoo[args.model](device)

    if model.input_shape == "(BATCH_SIZE, FRAME_LEN, D_MODEL)":
        flatten = False
    elif model.input_shape == "((BATCH_SIZE, FRAME_LEN), D_MODEL)":
        flatten = True
    else:
        raise ValueError(f"Unknown input_shape: {model.input_shape}")

    betas_source, betas_target = load_xy(args.betas_source, args.betas_target, device, flatten=flatten)
    
    if model.torch_or_numpy == "numpy":
        betas_source = betas_source.detach().cpu().numpy()
        betas_target = betas_target.detach().cpu().numpy()


    train_betas_source, test_betas_source, train_betas_target, test_betas_target = train_test_split(betas_source, betas_target, test_size=args.test_size)

    if isinstance(model, AutoSklearnRegressor):
        if isinstance(model.get_trials_callback, AutoSklearnCallback):
            model.get_trials_callback.setup(model, train_betas_source, train_betas_target, test_betas_source, test_betas_target, model_name)
    
    with autolog(args.model):
        train(model, train_betas_source, train_betas_target, test_betas_source, test_betas_target, output=args.output)

    if "show_models" in dir(model):
        print(model.show_models())
    if "leaderboard" in dir(model):
        print(model.leaderboard())
    if "get_models_with_weights" in dir(model):
        print(model.get_models_with_weights())
    if "runhistory_" in dir(model):
        with open(f"{model_name}_runhistory.pkl", "wb") as f:
            dill.dump(model.runhistory_, f)