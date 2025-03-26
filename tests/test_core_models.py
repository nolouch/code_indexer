import unittest
from core.models import CodeElement, Function, Class, Interface, Module, Variable, Annotation, CodeRepository

class TestCodeElement(unittest.TestCase):
    def test_code_element_creation(self):
        element = CodeElement(
            name="test_element",
            element_type="test",
            language="python",
            file="test.py",
            line=1
        )
        self.assertEqual(element.name, "test_element")
        self.assertEqual(element.element_type, "test")
        self.assertEqual(element.language, "python")
        self.assertEqual(element.file, "test.py")
        self.assertEqual(element.line, 1)
        self.assertEqual(element.docstring, "")
        self.assertEqual(element.attributes, {})

class TestFunction(unittest.TestCase):
    def test_function_creation(self):
        func = Function(
            name="test_func",
            element_type="function",
            language="python",
            file="test.py",
            line=1,
            parameters=[{"name": "arg1", "type": "str"}],
            return_types=["int"],
            calls=["other_func"],
            parent_class="TestClass"
        )
        self.assertEqual(func.name, "test_func")
        self.assertEqual(func.parameters, [{"name": "arg1", "type": "str"}])
        self.assertEqual(func.return_types, ["int"])
        self.assertEqual(func.calls, ["other_func"])
        self.assertEqual(func.parent_class, "TestClass")

class TestClass(unittest.TestCase):
    def test_class_creation(self):
        cls = Class(
            name="TestClass",
            element_type="class",
            language="python",
            file="test.py",
            line=1,
            fields=[{"name": "field1", "type": "str"}],
            methods=[],
            parent_classes=["BaseClass"],
            implements=["Interface1"]
        )
        self.assertEqual(cls.name, "TestClass")
        self.assertEqual(cls.fields, [{"name": "field1", "type": "str"}])
        self.assertEqual(cls.methods, [])
        self.assertEqual(cls.parent_classes, ["BaseClass"])
        self.assertEqual(cls.implements, ["Interface1"])

class TestModule(unittest.TestCase):
    def setUp(self):
        self.module = Module(
            name="test_module",
            element_type="module",
            language="python",
            file="test_module.py",
            line=1,
            imports=["os", "sys"],
            functions=[],
            classes=[],
            interfaces=[],
            submodules=["submodule1"],
            files=["test_module.py"]
        )

    def test_module_creation(self):
        self.assertEqual(self.module.name, "test_module")
        self.assertEqual(self.module.imports, ["os", "sys"])
        self.assertEqual(self.module.submodules, ["submodule1"])
        self.assertEqual(self.module.files, ["test_module.py"])

class TestCodeRepository(unittest.TestCase):
    def test_repository_creation(self):
        repo = CodeRepository(
            root_path="/test/repo",
            language="python",
            modules={},
            global_symbols={},
            metadata={}
        )
        self.assertEqual(repo.root_path, "/test/repo")
        self.assertEqual(repo.language, "python")
        self.assertEqual(repo.modules, {})
        self.assertEqual(repo.global_symbols, {})
        self.assertEqual(repo.metadata, {})

if __name__ == '__main__':
    unittest.main() 