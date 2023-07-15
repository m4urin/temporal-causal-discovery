from collections import defaultdict


def concat_dicts(my_dicts):
    merged_dict = defaultdict(list)
    for d in my_dicts:
        for key, value in d.items():
            merged_dict[key].append(value)
    return merged_dict
