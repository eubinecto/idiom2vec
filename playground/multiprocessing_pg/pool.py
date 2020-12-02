# This is a playground to get familiar with multiprocessing jobs
# code example  from: https://docs.python.org/3/library/multiprocessing.html
import multiprocessing as mp
import logging
from sys import stdout
logging.basicConfig(stream=stdout, level=logging.INFO)


def func(x):
    # the order in which this is computed is (should be?) arbitrary
    # well, but this does not have global access
    # use this pattern only if you don't need shared memory.
    print(x)


def main():
    with mp.Pool(processes=5) as p:
        batches = list(range(1000))
        # just provide batches to the function.
        # pool.map will apply function to each batch asynchronously.
        p.map(func, batches)


if __name__ == '__main__':
    main()
