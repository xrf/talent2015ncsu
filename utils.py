import subprocess, sys
import pandas as pd

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
    return pd.DataFrame.from_records(data)
