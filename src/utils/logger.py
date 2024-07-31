import sys

class PrintLogger:
    '''
    Simple logger class that can be used to print to stdout and log to a file at the same time.
    '''

    def __init__(self, filename=None, verbose=True):
        self.terminal = sys.stdout
        self.verbose = verbose
        self.log = open(filename, "w") if filename is not None else None

    def write(self, message):
        if self.verbose:
            self.terminal.write(message + '\n')
        if self.log is not None:
            self.log.write(message + '\n')

    def flush(self):
        pass

    def close(self):
        if self.log is not None:
            self.log.close()