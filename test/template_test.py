import pytest

from modules.template import Template

def testHasTemplate():
	assert Template().hasTemplate() == False
	assert Template(None).hasTemplate() == False
	assert Template("test/template.template").hasTemplate() == True

def testApplyTemplateNoTemplate1():
	template = Template()
	assert template.apply(input="abc") == "abc"

def testApplyTemplateNoTemplate3():
	template = Template()
	assert template.apply(input="abc", output="def", other="ghi") == "abcdefghi"

def testApplyTemplateInstructionOutput():
	template = Template("test/template.template")
	assert template.apply(instruction="do it", output="ok") == "Instruction.\n\n### Instruction:\ndo it\n\n### Response:\nok"

def testApplyTemplateInstructionInputOutput():
	template = Template("test/template.template")
	assert template.apply(instruction="do it", input="abc", output="ok") == "Input and instruction.\n\n### Instruction:\ndo it\n\n### Input:\nabc\n\n### Response:\nok"

def testApplyTemplateInvalidKey():
	template = Template("test/template.template")
	with pytest.raises(KeyError) as excinfo:
		template.apply(instruction="do it")
	assert str(excinfo.value) == "'instruction'"
