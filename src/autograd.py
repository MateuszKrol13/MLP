from abc import abstractmethod

VARIABLE_REGISTRY = set()

class Variable:
    def __init__(self, value, name):
        self.value=value
        self.name=name
        self.children=[]
        self._cache={}

    @abstractmethod
    def pass_gradient(self, grad):
        pass

class DifferentiableVariable(Variable):
    def __init__(self, value, name):
        self.children = []
        super().__init__(value, name)

    def __repr__(self):
        return f"DifferentiableVar(name={self.name}, val={self.value})"

    def register_parent(self, var):
        var.children.append(self)

