import os
import os.path as osp
import pyrootutils


__all__ = ['get_config_structure']


def get_config_structure(start_directory=None, indent=0, verbose=False): 
    """Parse a config file structure in search for .yaml files 
    """
    # If not provided, search the project configs directory
    if start_directory is None:
        root = str(pyrootutils.setup_root(
            search_from='',
            indicator=[".git", "README.md"],
            pythonpath=True,
            dotenv=True))
        start_directory = osp.join(root, 'configs')

    # Structure to store the file hierarchy: 
    #  - first value is a dictionary of directories
    #  - second value is a list of yaml files
    struct = ({}, [])

    # Recursively gather files and directories in the current directory
    for item in os.listdir(start_directory):
        item_path = os.path.join(start_directory, item)

        if os.path.isdir(item_path):
            if verbose:
                print(f"{'  ' * indent}Directory: {item}")
            struct[0][item] = get_config_structure(
                start_directory=item_path, indent=indent + 1)

        elif os.path.isfile(item_path):
            filename, extension = osp.splitext(item)
            if extension == '.yaml':
                struct[1].append(filename)
            if verbose:
                print(f"{'  ' * indent}File: {item}")

    return struct
