# This is a playground to get familiar with multiprocessing jobs
# code example  from: https://docs.python.org/3/library/multiprocessing.html
import multiprocessing as mp
import logging
from sys import stdout
logging.basicConfig(stream=stdout, level=logging.INFO)


def func(x):
    # the order in which this is computed is (should be?) arbitrary
    print(x)


def main():
    with mp.Pool(processes=5) as p:
        batches = list(range(1000))
        # jushttps://github.com/mdavolio/mancalat provide batches to the function.
        # pool.map will apply function to each batch asynchronously.
        p.map(func, batches)


if __name__ == '__main__':
    main()
