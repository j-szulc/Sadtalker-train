from inference import *

def train(model, train_betas_source, train_betas_target, test_betas_source, test_betas_target, output=None):
    model.fit(train_betas_source, train_betas_target)
    if output is not None:
        output = pathlib.Path(output).with_suffix(".pkl")
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output), "wb") as f:
            pickle.dump(model, f)
    train_score = model.score(train_betas_source, train_betas_target)
    print(f"Training score: {train_score}")
    if test_betas_source is not None:
        val_score = model.score(test_betas_source, test_betas_target)
        print(f"Validation score: {val_score}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas-target", type=pathlib.Path, nargs="+")
    parser.add_argument("--betas-source", type=pathlib.Path, nargs="+")
    parser.add_argument("--model", choices=zoo.keys(), default="automl")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    flatten = (args.model != "transformer")
    model = zoo[args.model](device)
    betas_source, betas_target = load_xy(args.betas_source, args.betas_target, flatten=flatten)
    train_betas_source, test_betas_source, train_betas_target, test_betas_target = train_test_split(betas_source, betas_target, test_size=args.test_size)
    train(model, train_betas_source, train_betas_target, test_betas_source, test_betas_target, output=args.output)

