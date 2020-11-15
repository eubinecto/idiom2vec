# This is a playground to get familiar with multiprocessing jobs
# code example  from: https://docs.python.org/3/library/multiprocessing.html
import multiprocessing as mp


def func(x):
    return x * x


def main():
    with mp.Pool(processes=3) as p:
        batches = [1, 2, 3]
        # just provide batches to the function.
        # pool.map will apply function to each batch asynchronously.
        print(p.map(func, batches))


if __name__ == '__main__':
    main()
