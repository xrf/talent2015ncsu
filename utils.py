import subprocess, sys
import pandas as pd

def flatten_dict(d):
    keys = tuple(d)
    for key in keys:
        if isinstance(d[key], dict):
            value = d.pop(key)
            for subkey, subvalue in value.items():
                d[key + "." + subkey] = subvalue

def run(*args, **kwargs):
    assert len(args) == 1
    name, = args
    p = subprocess.Popen(
        ["cargo", "run", "--release", "--bin=" + name, "--"] +
        sum((["--" + k.replace("_", "-"), str(v)]
             for k, v in kwargs.items()), []),
        stdout=subprocess.PIPE, universal_newlines=True)
    out = []
    for line in p.stdout:
        sys.stdout.write(line)
        out.append(line)
    out = "".join(out)
    data = eval(out)
    for row in data:
        flatten_dict(row)
    return pd.DataFrame.from_records(data)
