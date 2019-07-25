import json


class HyperParamSet:
    def __str__(self):
        json_obj = dict((attr, getattr(self, attr)) for attr in dir(self)
                        if hasattr(self, attr) and not attr.startswith('_'))
        return json.dumps(json_obj)