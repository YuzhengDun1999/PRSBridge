import numpy as np
from .tilted_stable import ExpTiltedStableDist

class BasicRandom():
    """
    Generators of random variables from the basic distributions used in
    Bayesian sparse regression.
    """

    def __init__(self, seed=None):
        self.np_random = np.random
        self.ts = ExpTiltedStableDist()
        self.set_seed(seed)

    def set_seed(self, seed):
        self.np_random.seed(seed)
        ts_seed = np.random.randint(1, 1 + np.iinfo(np.int32).max)
        self.ts.set_seed(ts_seed)

    def get_state(self):
        rand_gen_state = {
            'numpy' : self.np_random.get_state(),
            'tilted_stable' : self.ts.get_state(),
        }
        return rand_gen_state

    def set_state(self, rand_gen_state):
        self.np_random.set_state(rand_gen_state['numpy'])
        self.ts.set_state(rand_gen_state['tilted_stable'])

    def tilted_stable(self, char_exponent, tilt):
        return self.ts.sample(char_exponent, tilt)
