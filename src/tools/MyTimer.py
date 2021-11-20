import datetime


class MyTimer:
    def __init__(self):
        self.start = datetime.datetime.now()
        self.end = datetime.datetime.now()
        pass

    def stop(self):
        self.end = datetime.datetime.now()
        pass

    def total_time(self):
        return str(self.end - self.start)
        pass
    pass
