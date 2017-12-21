def print_shape(v):
    print(v.get_shape().as_list())

def epoch_to_step(epoch, total_size, batch_size):
    return int(total_size / batch_size * epoch)

def step_to_epoch(step, total_size, batch_size):
    return int(step * batch_size / total_size)


# https://stackoverflow.com/questions/4103773/efficient-way-of-having-a-function-only-execute-once-in-a-loop
# usage:
# @run_once
# def my_function(foo, bar):
#     return foo + bar
def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


