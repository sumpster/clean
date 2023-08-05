import re
import pytest

from modules.data import DataProcessor

basicPattern = re.compile(r'^in:dp(\d)\nout:data point (\d)$')
addPattern = re.compile(r'^in:dp(\d)\nadd:(\d)\nout:data point (\d)$')


def testLoadDataBasicOrdered():
    loadAndTestBasicData(randomize=False)

def testLoadDataBasicRandomized():
    loadAndTestBasicData(randomize=True)

def testLoadDataMixed():
    dp = DataProcessor("test/data.template")
    data = dp.loadData("test/data-mixed.json", randomize=False)
    assert len(data) == 5

    for i, entry in enumerate(data, start=1):
        if i == 3:
            assert addPattern.match(entry['input']), f"Row {i} does not match {addPattern}"
        else:
            assert basicPattern.match(entry['input']), f"Row {i} does not match {basicPattern}"


def loadAndTestBasicData(randomize):
    dp = DataProcessor("test/data.template")
    data = dp.loadData("test/data-basic.json", randomize=randomize, seed=42)

    assert len(data) == 5
    assert 'input' in data.column_names

    ordered = True
    ids = set()
    for i, entry in enumerate(data, start=1):
        m = basicPattern.match(entry['input'])
        assert m, f"Row {i} does not match {basicPattern}"
        assert m.group(1) == m.group(2)
        ordered &= int(m.group(1)) == i
        ids.add(int(m.group(1)))

    assert ordered != randomize, "If should be randomized but appears ordered, try different seed (and check for datasets / numpy rng changes)"
    assert len(ids) == 5
