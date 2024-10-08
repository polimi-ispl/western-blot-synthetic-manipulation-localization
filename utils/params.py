import os


def directory_find(atom, root='.'):

    list_dirs = []
    for path, dirs, files in os.walk(root):

        if atom in dirs and 'trainset' not in path:
            list_dirs.append(os.path.join(path, atom))

    return list_dirs


def path_find(atom, root='.'):

    list_dirs = []
    for path, dirs, files in os.walk(root):

        if atom in path:
            list_dirs.append(os.path.join(path))

    return list_dirs


################### FOLDERS
log_dir = 'runs_tampered_img_32'
models_dir = 'weights_tampered_img_32'
results_dir = 'results_realistically_tampered_img'

