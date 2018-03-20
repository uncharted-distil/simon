from os import listdir
from os.path import isfile, join


def put_dir(client, local_dir_path, destination_dir):
    files = [f for f in listdir(local_dir_path) if isfile(join(local_dir_path, f))]

    for f in files:
        client.put(join(local_dir_path, f), join(destination_dir, f))