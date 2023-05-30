import datetime


def now_str():
    """
    Returns:
        [str]: [returns with the string of datetime]
    """
    return datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")


def time_diff_str(start, end):
    """[Calculate the difference of 2 datetimes and format the strin]

    Args:
        start ([datetime]): [Start of period]
        end ([datetime]): [End of period]

    Returns:
        [str]: [String with the formatted time delta]
    """
    time_diff = (end - start).total_seconds()
    if time_diff > 3600:
        return f"{time_diff//3600:.0f} hours {(time_diff%3600)//60:.0f} minutes e {time_diff%60:.0f} seconds"
    elif time_diff > 60:
        return f"{time_diff//60:.0f} minutes e {time_diff%60:.0f} seconds"
    else:
        return f"{time_diff%60:.0f} seconds"


def now():
    """
    Returns:
        [str]: [returns with the datetime now]]
    """
    return datetime.datetime.now()


def timeit(func):
    def wrapper(*args, **kwargs):
        start = now()
        print(f"{now_str()} - Starting {func.__name__}...")
        result = func(*args, **kwargs)
        print(f"{now_str()} - Finished... Elapsed Time: {time_diff_str(start,now())}")
        return result

    return wrapper


def check_dict_keys(dict, keys):
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    if not all(key in dict for key in keys):
        raise AttributeError(f"dicts must contain the keys {', '.join(keys)}")
    return True
