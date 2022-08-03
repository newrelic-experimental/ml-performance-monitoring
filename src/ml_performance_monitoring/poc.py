def _patched_call(original_fn, patched_fn):
    def _inner_patch(*args, **kwargs):
        try:
            return patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex

    return _inner_patch

def test(x):
    print("Inside test")
    return "Y"

def patched_test(original_fn, *args, **kwargs):
    print(f"Before running {original_fn}. args: {args}; kwargs: {kwargs}")
    result = original_fn(*args, **kwargs)
    print(f"After running {original_fn}. result: {result}")
    return result

test = _patched_call(test, patched_test)

test("X")