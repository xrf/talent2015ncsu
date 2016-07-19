import subprocess, sys

def flatten_dict(d):
    '''Warning: this will modify the dictionary!'''
    keys = tuple(d)
    for key in keys:
        if isinstance(d[key], dict):
            value = d.pop(key)
            for subkey, subvalue in value.items():
                d[key + "." + subkey] = subvalue
    return d

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
    return eval(out)
