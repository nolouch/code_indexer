import unittest
from unittest.mock import Mock, patch
from semantic_graph.builder import SemanticGraphBuilder
from core.models import Module, Function, Class, Interface

class TestSemanticGraphBuilder(unittest.TestCase):
    def setUp(self):
        self.graph_mock = Mock()
        self.code_embedder_mock = Mock()
        self.doc_embedder_mock = Mock()
        
        self.builder = SemanticGraphBuilder(
            graph=self.graph_mock,
            code_embedder=self.code_embedder_mock,
            doc_embedder=self.doc_embedder_mock
        )

    def test_add_module(self):
        # Create test module
        module = Module(
            name="test_module",
            element_type="module",
            language="python",
            file="test_module.py",
            line=1,
            imports=["os"],
            functions=[
                Function(
                    name="test_func",
                    element_type="function",
                    language="python",
                    file="test_module.py",
                    line=10,
                    docstring="Test function"
                )
            ],
            classes=[
                Class(
                    name="TestClass",
                    element_type="class",
                    language="python",
                    file="test_module.py",
                    line=20,
                    docstring="Test class"
                )
            ],
            interfaces=[
                Interface(
                    name="TestInterface",
                    element_type="interface",
                    language="python",
                    file="test_module.py",
                    line=30,
                    docstring="Test interface"
                )
            ]
        )

        # Mock file content reading
        with patch("builtins.open", unittest.mock.mock_open(read_data="test content")):
            self.builder._add_module(module)

        # Print actual calls for debugging
        print("Actual calls:", self.graph_mock.add_node.mock_calls)
        
        # Verify graph node additions
        self.graph_mock.add_node.assert_any_call(
            module.name,
            type="module",
            language=module.language,
            file=module.file,
            code_content="test content"
        )

        # Verify function node
        self.graph_mock.add_node.assert_any_call(
            f"{module.name}.{module.functions[0].name}",
            type="function",
            language="python",
            file="test_module.py",
            line=10,
            docstring="Test function",
            code_content="test content"
        )

        # Verify class node
        self.graph_mock.add_node.assert_any_call(
            f"{module.name}.{module.classes[0].name}",
            type="class",
            language="python",
            file="test_module.py",
            line=20,
            docstring="Test class",
            code_content="test content"
        )

        # Verify interface node
        self.graph_mock.add_node.assert_any_call(
            f"{module.name}.{module.interfaces[0].name}",
            type="interface",
            language="python",
            file="test_module.py",
            line=30,
            docstring="Test interface",
            code_content="test content"
        )

        # Verify edges
        self.graph_mock.add_edge.assert_any_call(
            module.name,
            f"{module.name}.{module.functions[0].name}",
            type="contains"
        )
        self.graph_mock.add_edge.assert_any_call(
            module.name,
            f"{module.name}.{module.classes[0].name}",
            type="contains"
        )
        self.graph_mock.add_edge.assert_any_call(
            module.name,
            f"{module.name}.{module.interfaces[0].name}",
            type="contains"
        )

    def test_add_semantic_relations(self):
        # Mock graph nodes
        self.graph_mock.nodes.return_value = {
            "module.func1": {"type": "function"},
            "module.func2": {"type": "function"},
            "module.Class1": {"type": "class"},
            "module.Class2": {"type": "class"}
        }

        # Mock relation detection
        self.builder._has_call_relation = Mock(return_value=True)
        self.builder._has_inheritance_relation = Mock(return_value=True)
        self.builder._get_call_weight = Mock(return_value=1.0)
        self.builder._get_inheritance_weight = Mock(return_value=1.0)

        self.builder._add_semantic_relations()

        # Verify call relations
        self.graph_mock.add_edge.assert_any_call(
            "module.func1",
            "module.func2",
            type="calls",
            weight=1.0
        )

        # Verify inheritance relations
        self.graph_mock.add_edge.assert_any_call(
            "module.Class1",
            "module.Class2",
            type="inherits",
            weight=1.0
        )

    def test_add_embeddings(self):
        # Mock graph nodes
        self.graph_mock.nodes.return_value = {
            "test_node": {
                "code_content": "def test(): pass",
                "docstring": "Test function"
            }
        }

        # Mock embedders
        self.code_embedder_mock.embed.return_value = [0.1, 0.2, 0.3]
        self.doc_embedder_mock.embed.return_value = [0.4, 0.5, 0.6]

        self.builder._add_embeddings()

        # Verify embeddings were added
        self.code_embedder_mock.embed.assert_called_with("def test(): pass")
        self.doc_embedder_mock.embed.assert_called_with("Test function")

if __name__ == '__main__':
    unittest.main() 