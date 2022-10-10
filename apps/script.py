import subprocess


def start():
    cmd =['poetry', 'run', 'python',  'app.py']
    subprocess.run(cmd, shell=True)
