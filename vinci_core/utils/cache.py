cache = {}

def get_cached(prompt):
    return cache.get(prompt)

def set_cache(prompt, result):
    cache[prompt] = result
