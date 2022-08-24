from src.Interface.LoadSH import Gif48

def demo1():
    a = Gif48().load()
    C0, S0 = a.getCS(100)
    C1, S1 = a.getSigmaCS(100)
    print(a.getCS(100))
    pass


if __name__ == '__main__':
    demo1()

