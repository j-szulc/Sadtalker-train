from dataclasses import dataclass
from datetime import datetime, timedelta
import dill

@dataclass
class AutoSklearnCallback:
    counter: int = 0
    model_: object = None
    model_name: str = None
    X: object = None
    y: object = None
    X_test: object = None
    y_test: object = None
    last_counter_increase: datetime = datetime(1970, 1, 1)
    ckpt_every: timedelta = timedelta(minutes=10)

    def setup(self, model, X, y, X_test, y_test, model_name):
        self.model_ = model
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name

    def __call__(self, *_, **__):
        if datetime.now() - self.last_counter_increase >= self.ckpt_every:
            self.counter += 1
            self.last_counter_increase = datetime.now()
        try:
            train_score = self.model_.score(self.X, self.y)
            test_score = self.model_.score(self.X_test, self.y_test)
            print(datetime.now())
            print("train_score", train_score)
            print("test_score", test_score)
        except Exception as e:
            print(e)
            pass
        if "runhistory_" in dir(self.model_):
            with open(f"{self.model_name}_runhistory_{self.counter}.pkl", "wb") as f:
                dill.dump(self.model_.runhistory_, f)
        dill.dump(self.model_, open(f"{self.model_name}_checkpoint_{self.counter}.pkl", "wb"))
    
def get_automl(device):
    assert device == "cpu", "AutoML can only be run on CPU"
    from autosklearn.regression import AutoSklearnRegressor
    import autosklearn
    result = AutoSklearnRegressor(
        time_left_for_this_task=60*60*12,
        per_run_time_limit=60*6*12,
        memory_limit=30*1024,
        metric=autosklearn.metrics.mean_squared_error,
        n_jobs=1,
        get_trials_callback=AutoSklearnCallback()
    )
    result.input_shape = "((BATCH_SIZE, FRAME_LEN), D_MODEL)"
    result.torch_or_numpy = "numpy"
    return result