from torch import multiprocessing as mp

class RecRunner(object):
    def __init__(self,d):
        self.d = d
class MPERunner(RecRunner):
    def __init__(self,d):
        super(MPERunner, self).__init__(d)
        mp.set_start_method('spawn')
        p = mp.Process(target=self.new_processing_eval, args=(d,))
        p.start()
        p.join()

    def new_processing_eval(self, render=False):
        # while not self.stop_eval.value:
        #
        print("render:", render)

if __name__ == "__main__":
    s=MPERunner(True)

