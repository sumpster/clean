import pytest

from modules.template import Template

def testHasTemplate():
    assert Template().hasTemplate() == False
    assert Template(None).hasTemplate() == False
    assert Template("test/resources/data.template").hasTemplate() == True

def testApplyTemplateNoTemplate1():
    template = Template()
    assert template.apply(input="abc") == "abc"

def testApplyTemplateNoTemplate3():
    template = Template()
    assert template.apply(input="abc", output="def", other="ghi") == "abcdefghi"

def testApplyTemplateInputOutput():
    template = Template("test/resources/data.template")
    assert template.apply(input="do it", output="ok") == "in:do it\nout:ok"

def testApplyTemplateInputAddOutput():
    template = Template("test/resources/data.template")
    assert template.apply(input="do it", add="abc", output="ok") == "in:do it\nadd:abc\nout:ok"

def testApplyTemplateEmptyAdd():
    template = Template("test/resources/data.template")
    assert template.apply(input="do it", add="", output="ok") == "in:do it\nout:ok"

def testApplyTemplateEmptyOutput():
    template = Template("test/resources/data.template")
    assert template.apply(input="do it", add="", output="") == "in:do it\nout:"

def testApplyTemplateInvalidKey():
    template = Template("test/resources/data.template")
    with pytest.raises(KeyError) as excinfo:
        template.apply(instruction="do it")
    assert str(excinfo.value) == "'instruction'"
