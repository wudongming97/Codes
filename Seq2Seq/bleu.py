from os import system
from tempfile import NamedTemporaryFile


def bleu(reference, output):
    with NamedTemporaryFile('w+t', delete=False) as rf, NamedTemporaryFile('w+t', delete=False) as of:
        rf.write('\n'.join(reference))
        of.write('\n'.join(output))
    bleu_f(rf.name, of.name)


def bleu_f(rf, of):
    system('./multi-bleu.perl {} < {}'.format(rf, of))
