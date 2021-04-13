"""
cleanse with respect to each corpus.
"""
from typing import List, Callable
import re


class Cleaner:
    """
    Cleaner, applies to all domain.
    """
    def steps(self) -> List[Callable]:
        raise NotImplementedError

    def __call__(self, line: str) -> str:
        for step in self.steps():
            line = step(line)
        return line


class CocaCleaner(Cleaner):

    def steps(self) -> List[Callable]:
        return [
            self.remove_labels,
            self.replace_profanities
        ]

    @staticmethod
    def remove_labels(line: str) -> str:
        return re.sub(r'^@@[0-9]+? |^##[0-9]+? ', "", line).strip()

    @staticmethod
    def replace_profanities(line: str) -> str:
        return line.replace('@ @ @ @ @ @ @ @ @ @', 'PROFANITY').strip()


class CocaSpokCleaner(CocaCleaner):

    def steps(self) -> List[Callable]:
        return super(CocaSpokCleaner, self).steps() + [
            self.remove_speakers,
            self.remove_nonverbals
        ]

    @staticmethod
    def remove_speakers(line: str) -> str:
        """
        e.g. @!DAVID-GREENE#
        e.g. -MTP-DA#
        """
        return re.sub(r'@![a-zA-Z-]+# |@![a-zA-Z-]+ |[A-Z-]+# ',
                      "", line).strip()

    @staticmethod
    def remove_nonverbals(line: str) -> str:
        """
        they don't add much.
        e.g. 10 minutes ( ph ) have
        e.g. XI JINPING , CHINESE PRESIDENT ( through translator ) :
        e.g. Indeed it has become an irreversible historical trend . @ ( END VIDEO CLIP )
        """
        return re.sub(r'\( .+? \)', "", line).strip()


class CocaMagCleaner(CocaCleaner):

    def steps(self) -> List[Callable]:
        return super(CocaMagCleaner, self).steps() + [
            self.remove_sections,
            self.remove_p_tags
        ]

    @staticmethod
    def remove_sections(line: str) -> str:
        """
        e.g. ##2018807 Section : INVESTING <p> No one can predict a s
        """
        return re.sub(r'Section : [A-Z\s]+? ', "", line).strip()

    @staticmethod
    def remove_p_tags(line: str) -> str:
        return line.replace("<p>", "").strip()


class OpenSubCleaner(Cleaner):
    def steps(self) -> List[Callable]:
        return [
            self.clean
        ]

    @staticmethod
    def clean(line: str) -> str:
        return line   # just pass it.