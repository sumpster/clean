import re
import json

class Template:
    def __init__(self, templatePath = None):
        if templatePath:
            with open(templatePath, 'r') as file:
                self.templates = self._createTemplateMap(json.load(file))
        else:
            self.templates = None


    def _createTemplateMap(self, data):
        result = {}

        for element in data:
            placeholders = sorted(re.findall(r'\{(\w+)\}', element))
            key = ','.join(placeholders)
            result[key] = element

        return result


    def apply(self, **kwargs):
        fields = sorted([field for field in kwargs if kwargs[field] or field == "output"])
        key = ','.join(fields)

        if not self.templates:
            if len(fields) != 2:
                raise KeyError(f"Without template there must be exactly one input and one output field. Found: {fields}")
            return kwargs[fields[0]]

        template = self.templates[key]
        format_args = { field: kwargs[field] for field in fields }
        return template.format(**format_args)


    def hasTemplate(self):
        return self.templates != None
