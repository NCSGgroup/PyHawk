import pathlib as pathlib





class PathOfFiles:
    root_dir = pathlib.Path(__file__).parent.parent.parent
    NormalEqTemp = root_dir.joinpath('temp', 'NEQ/')
