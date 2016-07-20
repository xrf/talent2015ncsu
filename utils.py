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

def run(name, *args, quiet=False):
    cmd = ["cargo", "run", "--release", "--bin=" + name, "--"]
    cmd.extend(args)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    out = []
    for line in p.stdout:
        if not quiet or line.startswith("#"):
            sys.stdout.write(line)
        out.append(line)
    p.wait()
    if p.returncode:
        exit(p.returncode)
    return eval("".join(out))
