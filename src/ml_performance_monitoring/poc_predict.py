from import_bind import PostImportHookPatching

def _patched_call(original_fn, patched_fn):
    def _inner_patch(*args, **kwargs):
        try:
            return patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex

    return _inner_patch

def patcher_predict(original_fn, self, X, check_input=True):
        print(f"Before running {original_fn}. X: {X}; check_input: {check_input}")
        result = original_fn(self, X, check_input)
        print(f"After running {original_fn}. result: {result}")
        return result

### Patch if not imported yet
# from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

def perform_patch():
    print("perform_patch")
    DecisionTreeRegressor.predict = _patched_call(DecisionTreeRegressor.predict, patcher_predict)

perform_patch()
