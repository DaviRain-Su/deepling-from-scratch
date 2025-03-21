class Man:
    def __init__(self, name):
        self.name = name
        print("Initiated!")

    def hello(self):
        print(f"Hello, {self.name}!")

    def goodbye(self):
        print(f"Goodbye, {self.name}!")


me = Man("John")
me.hello()
me.goodbye()
