from config.core import config
from evaluation.eval import *
import pytest



def test_generator_eval():
    generated_response="NRMA insurance has a maximum liability cover of 20 Million dollars"
    answer="Max liability cover for NRMA is $20,000,000"
    similarity=generator_eval(generated_response,answer)

    assert similarity>.5