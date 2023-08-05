import pytest

from modules.settings import Settings

def testBasePath():
    settings = Settings('test/resources/settings-min.json')
    assert settings.base.path == "model"

def testTemplatePath():
    settings = Settings('test/resources/settings-min.json')
    assert settings.templatePath == "test.template"

def testFallbackTemplatePath():
    settings = Settings('test/resources/settings-min.json')
    assert settings.inference.templatePath == "test.template"
    assert settings.training.templatePath == "test.template"

def testExplicitTemplatePath():
    settings = Settings('test/resources/settings.json')
    assert settings.inference.templatePath == "inference.template"
    assert settings.training.templatePath == "training.template"

def testFallbackOutputPath():
    settings = Settings('test/resources/settings-min.json')
    assert settings.training.outputPath == "test/resources/settings-min"

def testSectionDefaults():
    settings = Settings('test/resources/settings-min.json')
    assert settings.base.bits == 8
    assert settings.adapter.loraR == 16
    assert settings.training.cutoff == 256
    assert settings.inference.maxLength == 1024
    assert settings.ui.title == ""

def testSectionParsing():
    settings = Settings('test/resources/settings.json')
    assert settings.base.path == "mymodel"
    assert settings.adapter.loraR == 40
    assert settings.training.cutoff == 1024
    assert settings.inference.maxLength == 2048
    assert settings.ui.title == "Test"