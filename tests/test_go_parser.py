import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock
from parsers.go.parser import GoParser
from core.models import Module, Function, Class, Interface

class TestGoParser(unittest.TestCase):
    def setUp(self):
        self.parser = GoParser()
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test.go")
        
        # Sample Go code for testing
        self.test_code = """
package main

import (
    "fmt"
    "strings"
)

// Person represents a person with basic information
type Person struct {
    Name    string `json:"name"`
    Age     int    `json:"age"`
    Address string `json:"address"`
}

// Greeter defines the greeting behavior
type Greeter interface {
    Greet() string
    SetName(name string)
}

// Implement Greeter interface for Person
func (p *Person) Greet() string {
    return fmt.Sprintf("Hello, my name is %s and I am %d years old", p.Name, p.Age)
}

func (p *Person) SetName(name string) {
    p.Name = strings.TrimSpace(name)
}

// CreatePerson creates a new person with the given information
func CreatePerson(name string, age int, address string) *Person {
    return &Person{
        Name:    name,
        Age:     age,
        Address: address,
    }
}
"""
        # Write test code to file
        with open(self.test_file, "w") as f:
            f.write(self.test_code)

    def tearDown(self):
        # Clean up test directory
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    @patch("subprocess.run")
    def test_setup(self, mock_run):
        # Mock successful build
        mock_run.return_value = Mock(returncode=0)
        
        self.parser.setup()
        
        # Verify AST analyzer was built
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "go")
        self.assertEqual(args[1], "build")

    @patch("subprocess.run")
    def test_parse_file(self, mock_run):
        # Mock AST analyzer output with valid JSON
        mock_run.return_value = Mock(
            stdout='{"main":{"name":"main","path":"main","files":["test.go"],"imports":["fmt","strings"],"functions":[{"name":"CreatePerson","package":"main","file":"test.go","line":31,"docstring":"CreatePerson creates a new person with the given information","calls":[]}],"structs":[{"name":"Person","package":"main","file":"test.go","line":9,"docstring":"Person represents a person with basic information","fields":[{"name":"Name","type":"string","tag":"json:\\"name\\""},{"name":"Age","type":"int","tag":"json:\\"age\\""},{"name":"Address","type":"string","tag":"json:\\"address\\""}],"methods":["Greet","SetName"]}],"interfaces":[{"name":"Greeter","package":"main","file":"test.go","line":16,"docstring":"Greeter defines the greeting behavior","methods":["Greet","SetName"]}]}}',
            returncode=0
        )

        # Test with default parameters
        module = self.parser.parse_file(self.test_file)
        self.assertEqual(module.name, "main")
        
        # Create a new mock for testing with specific parameters
        specific_mock = Mock(
            stdout='{"main":{"name":"main","path":"main","files":["test.go"],"imports":["fmt","strings"],"functions":[{"name":"CreatePerson","package":"main","file":"test.go","line":31,"docstring":"CreatePerson creates a new person with the given information","calls":[]}],"structs":[{"name":"Person","package":"main","file":"test.go","line":9,"docstring":"Person represents a person with basic information","fields":[{"name":"Name","type":"string","tag":"json:\\"name\\""},{"name":"Age","type":"int","tag":"json:\\"age\\""},{"name":"Address","type":"string","tag":"json:\\"address\\""}],"methods":["Greet","SetName"]}],"interfaces":[{"name":"Greeter","package":"main","file":"test.go","line":16,"docstring":"Greeter defines the greeting behavior","methods":["Greet","SetName"]}]}}',
            returncode=0
        )
        mock_run.return_value = specific_mock
        
        # Test with specific parameters
        module = self.parser.parse_file(self.test_file, specific_package="main", exclude_tests=False)

        # Verify module
        self.assertEqual(module.name, "main")
        self.assertEqual(module.language, "go")
        self.assertEqual(module.imports, ["fmt", "strings"])
        
        # Verify function
        self.assertEqual(len(module.functions), 1)
        func = module.functions[0]
        self.assertEqual(func.name, "CreatePerson")
        self.assertEqual(func.docstring, "CreatePerson creates a new person with the given information")
        
        # Verify struct
        self.assertEqual(len(module.classes), 1)
        struct = module.classes[0]
        self.assertEqual(struct.name, "Person")
        self.assertEqual(struct.docstring, "Person represents a person with basic information")
        self.assertEqual(len(struct.fields), 3)
        
        # Verify interface
        self.assertEqual(len(module.interfaces), 1)
        iface = module.interfaces[0]
        self.assertEqual(iface.name, "Greeter")
        self.assertEqual(iface.docstring, "Greeter defines the greeting behavior")
        self.assertEqual(len(iface.methods), 2)
        
        # Verify that correct command parameters were passed
        args = mock_run.call_args[0][0]
        self.assertIn("-package", args)
        self.assertIn("main", args)
        self.assertIn("-exclude-tests=false", args)

    def test_get_dependencies(self):
        # Create test module with dependencies
        module = Module(
            name="main",
            element_type="module",
            language="go",
            file="test.go",
            line=1,
            imports=["fmt", "strings"],
            functions=[
                Function(
                    name="CreatePerson",
                    element_type="function",
                    language="go",
                    file="test.go",
                    line=31,
                    calls=["fmt.Sprintf"]
                )
            ],
            classes=[
                Class(
                    name="Person",
                    element_type="struct",
                    language="go",
                    file="test.go",
                    line=9,
                    implements=["Greeter"]
                )
            ]
        )

        deps = self.parser.get_dependencies(module)
        
        # Verify import dependencies
        self.assertIn("fmt", [d.name for d in deps["imports"]])
        self.assertIn("strings", [d.name for d in deps["imports"]])
        
        # Verify function call dependencies
        func = module.functions[0]
        func_deps = self.parser.get_dependencies(func)
        self.assertIn("fmt.Sprintf", [d.name for d in func_deps["calls"]])
        
        # Verify interface implementation dependencies
        struct = module.classes[0]
        struct_deps = self.parser.get_dependencies(struct)
        self.assertIn("Greeter", [d.name for d in struct_deps["implements"]])

    @patch("subprocess.run")
    @patch("parsers.go.parser.GoParser._get_packages")
    def test_parse_directory(self, mock_get_packages, mock_run):
        # Mock packages list
        mock_get_packages.return_value = [self.test_dir]
        
        # Mock AST analyzer output
        mock_run.return_value = Mock(
            stdout='{"main":{"name":"main","path":"main","files":["test.go"],"imports":["fmt","strings"],"functions":[{"name":"CreatePerson","package":"main","file":"test.go","line":31,"docstring":"CreatePerson creates a new person with the given information","calls":[]}],"structs":[{"name":"Person","package":"main","file":"test.go","line":9,"docstring":"Person represents a person with basic information","fields":[{"name":"Name","type":"string","tag":"json:\\"name\\""},{"name":"Age","type":"int","tag":"json:\\"age\\""},{"name":"Address","type":"string","tag":"json:\\"address\\""}],"methods":["Greet","SetName"]}],"interfaces":[{"name":"Greeter","package":"main","file":"test.go","line":16,"docstring":"Greeter defines the greeting behavior","methods":["Greet","SetName"]}]}}',
            returncode=0
        )
        
        # Test with default parameters
        repo = self.parser.parse_directory(self.test_dir)
        self.assertIsNotNone(repo)
        self.assertEqual(len(repo.modules), 1)
        
        # Test with specific parameters
        repo = self.parser.parse_directory(self.test_dir, specific_package="main", exclude_tests=False)
        self.assertIsNotNone(repo)
        self.assertEqual(len(repo.modules), 1)
        
        # Verify correct command parameters were passed
        args = mock_run.call_args[0][0]
        self.assertIn("-package", args)
        self.assertIn("main", args)
        self.assertIn("-exclude-tests=false", args)

if __name__ == '__main__':
    unittest.main() 