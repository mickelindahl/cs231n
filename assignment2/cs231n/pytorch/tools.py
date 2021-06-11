import hashlib


def string_to_hash(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()
