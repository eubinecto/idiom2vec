# code modified from: https://github.com/PolyAI-LDN/conversational-datasets/blob/master/opensubtitles/create_data.py
import re
import argparse
from typing import Generator
import json
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# some default parameters
MIN_LENGTH = 6
MAX_LENGTH = 127
NUM_EXTRA_CONTEXTS = 10


def _should_skip(line, min_length, max_length):
    """Whether a line should be skipped depending on the length."""
    return len(line) < min_length or len(line) > max_length


def create_example(previous_lines, line) -> dict:
    """Creates examples with multi-line context
    The examples will include:
        response: the current line text
        context: the previous line text
        context/0: 2 lines before
        context/1: 3 lines before, etc.
    """
    example = {
        'context': previous_lines[-1],
        'response': line
    }

    extra_contexts = previous_lines[:-1]
    example.update({
        'context/{}'.format(i): context
        for i, context in enumerate(extra_contexts[::-1])
    })

    return example


def _preprocess_line(line) -> str:
    # Remove the first word if it is followed by colon (speaker names)
    # NOTE: this wont work if the speaker's name has more than one word
    line = re.sub('(?:^|(?:[.!?]\\s))(\\w+):', "", line)

    # Remove anything between brackets (corresponds to acoustic events).
    line = re.sub("[\\[(](.*?)[\\])]", "", line)

    # Strip blanks hyphens and line breaks
    line = line.strip(" -\n")

    return line


def _create_examples_from_file(file_path, min_length, max_length,
                               num_extra_contexts) -> Generator[dict, None, None]:
    previous_lines = []
    # read in the text file
    with open(file_path, 'r') as fh:
        for line in fh:
            line = _preprocess_line(line)
            if not line:
                continue

            should_skip = _should_skip(
                line,
                min_length=min_length,
                max_length=max_length)

            if previous_lines:
                should_skip |= _should_skip(
                    previous_lines[-1],
                    min_length=min_length,
                    max_length=max_length)

                if not should_skip:
                    yield create_example(previous_lines, line)

            previous_lines.append(line)
            if len(previous_lines) > num_extra_contexts + 1:
                del previous_lines[0]


def main():
    global MIN_LENGTH, MAX_LENGTH, NUM_EXTRA_CONTEXTS
    logger = logging.getLogger("main")
    # path to en.txt
    # path to en.ndjson
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "en_txt_path",
        type=str,
        help="the path to en.txt")
    parser.add_argument(
        "en_ndjson_path",
        type=str,
        help="the path to en.ndjson")
    # parse the args and get the paths
    args = parser.parse_args()
    en_txt_path = args.en_txt_path
    en_ndjson_path = args.en_ndjson_path
    # get the total number of lines
    with open(en_txt_path, 'r') as fh:
        total = sum((1 for _ in fh))

    # start writing to it
    with open(en_ndjson_path, 'w') as fh:
        examples = _create_examples_from_file(en_txt_path, MIN_LENGTH,
                                              MAX_LENGTH, NUM_EXTRA_CONTEXTS)
        for idx, example in enumerate(examples):
            fh.write(json.dumps(example))
            # new-line delimited
            fh.write("\n")
            logger.info("{}/{}".format(idx + 1, total))


if __name__ == '__main__':
    main()
