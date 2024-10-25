from src.Interface.LoadSH import LoadGif48, LoadGLO


def demo1():
    a = LoadGif48().load()
    C0, S0 = a.getCS(100)
    C1, S1 = a.getSigmaCS(100)
    print(a.getCS(100))
    pass


def demo2():
    FileDir = "../data/AOD/RL06"
    AOD_type = "GLO"
    ld = LoadGLO().load(FileDir).setType(kind=AOD_type)


if __name__ == '__main__':
    demo1()

