class Logger:
    @staticmethod
    def info(*msg):
        print(f"[INFO]: {msg}")

    @staticmethod
    def debug(msg, **kwargs):
        print(f"[DEBUG]: {msg} ; {kwargs=} ")
