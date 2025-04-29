from __future__ import annotations

import copy

import pytest
import numpy as np
import networkx as nx

from semantic_kg import perturbation


@pytest.fixture
def test_di_graph() -> nx.DiGraph:
    # Create a simple graph
    graph = nx.DiGraph()

    graph.add_node("A", id="A", node_type="allowed")
    graph.add_node("B", id="B", node_type="allowed")
    graph.add_node("C", id="C", node_type="allowed")
    graph.add_node("D", id="D", node_type="allowed")
    graph.add_node("E", id="E", node_type="allowed")
    graph.add_node("F", id="F", node_type="allowed")
    graph.add_node("G", id="G", node_type="allowed")

    graph.add_edge("A", "B", effect="increase")
    graph.add_edge("A", "C", effect="increase")
    graph.add_edge("A", "D", effect="increase")
    graph.add_edge("A", "E", effect="increase")
    graph.add_edge("B", "F", effect="increase")
    graph.add_edge("B", "G", effect="increase")

    return graph


class TestEdgeAdditionPerturbation:
    @pytest.fixture(scope="class")
    def test_graph(self) -> nx.Graph:
        # Create a simple graph
        graph = nx.Graph()

        graph.add_node("A", id="A", node_type="A")
        graph.add_node("B", id="B", node_type="B")
        graph.add_node("C", id="C", node_type="C")

        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        return graph

    def test_create(self, test_graph: nx.Graph) -> None:
        perturber = perturbation.EdgeAdditionPerturbation(node_type_field="node_type")
        p_graph = perturber.create(test_graph)

        assert p_graph.number_of_edges() == test_graph.number_of_edges() + 1
        assert "C" in list(p_graph.neighbors("A"))
        assert perturber.edit_count == 1

        # Check perturbation is correctly logged
        assert perturber.perturbation_log[-1]["type"] == "edge_addition"
        assert perturber.perturbation_log[-1]["source"] in ["A", "C"]
        assert perturber.perturbation_log[-1]["target"] in ["A", "C"]
        assert perturber.perturbation_log[-1]["metadata"] == {}

    def test_create_valid_edges(self, test_graph: nx.Graph) -> None:
        """Only allow edges that already exist which should raise an error"""
        # Existing edges are only valid edges
        valid_node_pairs = [("A", "B"), ("B", "C")]

        perturber = perturbation.EdgeAdditionPerturbation(
            valid_node_pairs=valid_node_pairs
        )
        with pytest.raises(perturbation.NoValidEdgeError):
            _ = perturber.create(test_graph)

    def test_create_edge_map(self, test_graph: nx.Graph) -> None:
        # Type annotation required to avoid Literal[str] error
        edge_map: dict[tuple[str, str], list[str]] = {
            ("A", "C"): ["mock_name"],
            ("C", "A"): ["mock_name"],
        }

        perturber = perturbation.EdgeAdditionPerturbation(edge_map=edge_map)
        p_graph = perturber.create(test_graph)

        assert p_graph.number_of_edges() == test_graph.number_of_edges() + 1
        assert "C" in list(p_graph.neighbors("A"))
        assert perturber.edit_count == 1
        assert p_graph["A"]["C"]["edge_name"] == "mock_name"

    def test_reset(self, test_graph: nx.Graph) -> None:
        perturber = perturbation.EdgeAdditionPerturbation(node_type_field="id")
        _ = perturber.create(test_graph)

        perturber.reset()
        assert perturber.edit_count == 0
        assert perturber.perturbation_log == []

    def test_create_directed(self, test_di_graph: nx.DiGraph) -> None:
        perturber = perturbation.EdgeAdditionPerturbation(node_type_field="id")
        p_graph = perturber.create(test_di_graph)
        assert isinstance(p_graph, nx.DiGraph)


class TestEdgeDeletionPerturbation:
    @pytest.fixture(scope="class")
    def test_graph(self) -> nx.Graph:
        """Creates a test graph where only A and B have > 1 edges"""
        graph = nx.Graph()

        graph.add_node("A", id="A")
        graph.add_node("B", id="B")
        graph.add_node("C", id="C")
        graph.add_node("D", id="D")
        graph.add_node("E", id="E")
        graph.add_node("F", id="F")
        graph.add_node("G", id="G")

        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("A", "D")
        graph.add_edge("A", "E")
        graph.add_edge("B", "F")
        graph.add_edge("B", "G")

        return graph

    def test_create(self, test_graph: nx.Graph) -> None:
        perturber = perturbation.EdgeDeletionPerturbation(directed=False)
        p_graph = perturber.create(test_graph)

        assert p_graph.number_of_edges() == test_graph.number_of_edges() - 1
        assert "B" not in list(p_graph.neighbors("A"))
        assert perturber.edit_count == 1

    def test_create_perturbation_log(self, test_graph: nx.Graph) -> None:
        perturber = perturbation.EdgeDeletionPerturbation(directed=False)
        _ = perturber.create(test_graph)

        assert len(perturber.perturbation_log) == 1
        assert perturber.perturbation_log[0]["type"] == "edge_deletion"
        assert perturber.perturbation_log[0]["source"] in ["A", "B"]
        assert perturber.perturbation_log[0]["target"] in ["A", "B"]

    def test_invalid_create(self, test_graph: nx.Graph) -> None:
        perturber = perturbation.EdgeDeletionPerturbation(directed=False)
        p_graph = perturber.create(test_graph)

        # Once A-B edge deleted, no edges are valid for deletion
        with pytest.raises(perturbation.NoValidEdgeError):
            _ = perturber.create(p_graph)

    def test_create_directed(self, test_di_graph: nx.DiGraph) -> None:
        perturber = perturbation.EdgeDeletionPerturbation()
        p_graph = perturber.create(test_di_graph)
        assert isinstance(p_graph, nx.DiGraph)


class TestEdgeReplacementPerturbation:
    @pytest.fixture(scope="class")
    def test_graph(self) -> nx.Graph:
        # Create a simple graph
        graph = nx.Graph()

        graph.add_node("A", id="A", node_type="mock")
        graph.add_node("B", id="B", node_type="mock")
        graph.add_node("C", id="C", node_type="mock")

        graph.add_edge("A", "B", edge_name="increase")
        graph.add_edge("B", "C", edge_name="decrease")

        return graph

    def test_create(
        self,
        test_graph: nx.Graph,
    ) -> None:
        perturber = perturbation.EdgeReplacementPerturbation(
            replace_map={"increase": ["decrease"], "decrease": ["increase"]},
            directed=False,
        )

        p_graph = perturber.create(test_graph)

        assert p_graph.number_of_edges() == test_graph.number_of_edges()
        # Find the replaced edge
        replaced_edge = None
        for u, v, data in test_graph.edges(data=True):
            if data["edge_name"] != p_graph.get_edge_data(u, v)["edge_name"]:
                replaced_edge = (u, v)

        # Checks a replaced edge is found
        assert replaced_edge is not None
        # Checks the edge `effect` attribute has been changed
        assert (
            p_graph.get_edge_data(*replaced_edge)["edge_name"]
            != test_graph.get_edge_data(*replaced_edge)["edge_name"]
        )

        assert perturber.edit_count == 1

        # Check log is correctly updated
        assert perturber.perturbation_log[-1]["type"] == "edge_replacement"
        assert perturber.perturbation_log[-1]["source"] in list(replaced_edge)
        assert perturber.perturbation_log[-1]["target"] in list(replaced_edge)
        assert perturber.perturbation_log[-1]["metadata"] == {
            "edge_name": p_graph.get_edge_data(*replaced_edge)["edge_name"]
        }

    def test_create_invalid(
        self,
        test_graph: nx.Graph,
    ) -> None:
        modified_graph = test_graph.copy()

        # `replace_map` set to contain edge-values not found in current graph
        perturber = perturbation.EdgeReplacementPerturbation(
            replace_map={"elevate": ["reduce"], "reduce": ["elevate"]},
            directed=False,
        )

        with pytest.raises(perturbation.NoValidEdgeError):
            _ = perturber.create(modified_graph)

    def test_create_directed(
        self,
        test_graph: nx.Graph,
    ) -> None:
        perturber = perturbation.EdgeReplacementPerturbation(
            replace_map={"increase": ["decrease"], "decrease": ["increase"]},
            directed=True,
        )
        p_graph = perturber.create(test_graph)
        assert isinstance(p_graph, nx.DiGraph)


class TestNodeRemovalPerturbation:
    @pytest.fixture(scope="class")
    def test_graph(self) -> nx.Graph:
        # Create a simple graph
        graph = nx.Graph()

        graph.add_node("A", id="A")
        graph.add_node("B", id="B")
        graph.add_node("C", id="C")
        graph.add_node("D", id="D")

        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("B", "D")
        graph.add_edge("C", "D")

        return graph

    def test_create(self, test_graph: nx.Graph) -> None:
        # Number of edits should be equal to the number of nodes + edges removed
        expected_edit_map = {
            "A": 2,
            "B": 5,
            "C": 3,
            "D": 3,
        }

        perturber = perturbation.NodeRemovalPerturbation()
        p_graph = perturber.create(test_graph)

        # Find missing node
        missing_node = None
        for node in test_graph.nodes():
            if node not in p_graph.nodes():
                missing_node = node

        assert missing_node is not None
        assert missing_node not in p_graph.nodes()

        # Check edit count is as expected
        assert perturber.edit_count == expected_edit_map[missing_node]

        # Check log is correctly updated
        assert perturber.perturbation_log[-1]["type"] == "node_removal"
        assert perturber.perturbation_log[-1]["source"] == missing_node
        assert perturber.perturbation_log[-1]["target"] is None
        assert (
            perturber.perturbation_log[-1]["metadata"] == test_graph.nodes[missing_node]
        )

    def test_create_cleanup(self, test_graph: nx.Graph) -> None:
        """Checks that any degree 0 nodes removed"""
        perturber = perturbation.NodeRemovalPerturbation()
        p_graph = perturber.create(test_graph, "B")

        assert "B" not in p_graph.nodes()
        # A becomes degree 0 after removal so should be removed
        assert "A" not in p_graph.nodes()
        assert perturber.edit_count == 5

    def test_create_invalid(self) -> None:
        """Tests that a node can't be removed if all nodes deleted"""
        graph = nx.Graph()

        graph.add_node("A", id="A")
        graph.add_node("B", id="B")
        graph.add_node("C", id="C")
        graph.add_node("D", id="D")

        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("A", "D")

        perturber = perturbation.NodeRemovalPerturbation()
        with pytest.raises(perturbation.NoValidNodeError):
            _ = perturber.create(graph, "A")

    def test_create_directed(self, test_di_graph: nx.DiGraph) -> None:
        perturber = perturbation.NodeRemovalPerturbation()
        p_graph = perturber.create(test_di_graph)
        assert isinstance(p_graph, nx.DiGraph)


class TestNodeReplacementPerturbation:
    @pytest.fixture(scope="class")
    def test_graph(self) -> nx.Graph:
        # Create a simple graph
        graph = nx.Graph()

        graph.add_node("A", node_name="A", node_type="type_1")
        graph.add_node("B", node_name="B", node_type="type_2")
        graph.add_node("C", node_name="C", node_type="type_2")

        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        return graph

    @pytest.fixture(scope="class")
    def replace_map(self) -> dict[str, dict[str, list[str]]]:
        return {
            "node_type": {
                "type_1": ["A", "B", "D", "E", "F"],
                "type_2": ["C", "G"],
            }
        }

    def test_init_invalid_input(
        self, replace_map: dict[str, dict[str, list[str]]]
    ) -> None:
        # Check an error raised when neither `replace_opts` or `replace_map` specified
        with pytest.raises(ValueError):
            _ = perturbation.NodeReplacementPerturbation(node_attr_field="node_name")

        # Check an error raised when both `replace_opts` and `replace_map` specified
        with pytest.raises(ValueError):
            _ = perturbation.NodeReplacementPerturbation(
                node_attr_field="node_name",
                replace_opts=["B", "C"],
                replace_map=replace_map,
            )

        # Checks an error raised if `replace_map` conditions on multiple node attributes
        extra_replace_map = copy.deepcopy(replace_map)
        extra_replace_map["node_name"] = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
        }
        with pytest.raises(ValueError):
            _ = perturbation.NodeReplacementPerturbation(
                node_attr_field="node_name", replace_map=extra_replace_map
            )

    def test_create_replace_map(
        self, test_graph: nx.Graph, replace_map: dict[str, dict[str, list[str]]]
    ) -> None:
        perturber = perturbation.NodeReplacementPerturbation(
            node_attr_field="node_name",
            replace_map=replace_map,
        )
        p_graph = perturber.create(test_graph, replace_node="A")

        node_names = [p_graph.nodes[n]["node_name"] for n in p_graph.nodes]
        assert "A" not in node_names

        # Find the node that was replaced (should be "type_1")
        replace_node = None
        for node in p_graph.nodes:
            if p_graph.nodes[node]["node_type"] == "type_1":
                replace_node = node

        assert replace_node
        # Checks that it was replaced with one of the types from the dict
        assert (
            p_graph.nodes[replace_node]["node_name"]
            in replace_map["node_type"]["type_1"]
        )

    def test_create_replace_opts(self, test_graph: nx.Graph) -> None:
        replace_opts = ["A", "B", "C"]
        perturber = perturbation.NodeReplacementPerturbation(
            node_attr_field="node_name",
            replace_opts=replace_opts,
        )
        p_graph = perturber.create(test_graph, replace_node="A")

        node_names = [p_graph.nodes[n]["node_name"] for n in p_graph.nodes]
        assert "A" not in node_names

        # Find the node that was replaced (should be "type_1")
        replace_node = None
        for node in p_graph.nodes:
            if p_graph.nodes[node]["node_type"] == "type_1":
                replace_node = node

        assert replace_node
        # Checks that it was replaced with one of the types from the list
        assert p_graph.nodes[replace_node]["node_name"] in replace_opts

    def test_create_no_new_name(self, test_graph: nx.Graph) -> None:
        # Only allow for replacement of same-type as current node
        node_type_opts = {"node_type": {"type_1": ["A"]}}

        perturber = perturbation.NodeReplacementPerturbation(
            node_attr_field="node_name", replace_map=node_type_opts
        )
        with pytest.raises(perturbation.NoValidNodeError):
            _ = perturber.create(test_graph, replace_node="A")

        # Checks error also raised for `replace_opts`
        perturber = perturbation.NodeReplacementPerturbation(
            node_attr_field="node_name", replace_opts=["A"]
        )
        with pytest.raises(perturbation.NoValidNodeError):
            _ = perturber.create(test_graph, replace_node="A")

    def test_create_no_replace_node(
        self,
        test_graph: nx.Graph,
        replace_map: dict[str, dict[str, list[str]]],
    ) -> None:
        perturber = perturbation.NodeReplacementPerturbation(
            node_attr_field="node_name", replace_map=replace_map
        )
        p_graph = perturber.create(test_graph)

        node_names = [p_graph.nodes[n]["node_name"] for n in p_graph.nodes]
        assert node_names != ["A", "B", "C"]


class MockPerturbation(perturbation.BasePerturbation):
    def __init__(self):
        super().__init__()
        self.n_perturbations = 0

    def create(self, graph: nx.Graph) -> nx.Graph:
        p_graph = nx.Graph(graph)

        if self.n_perturbations == 0:
            # Remove B -> C edge
            p_graph.remove_edge("B", "C")
            self.n_perturbations += 1
            self._log_perturbation("edge_deletion", "B", "C")
            return p_graph
        elif self.n_perturbations == 1:
            # Add A -> C edge
            p_graph.add_edge("A", "C", id="increase")
            self.n_perturbations += 1
            self._log_perturbation("edge_addition", "A", "C")
            return p_graph
        else:
            raise perturbation.NoValidEdgeError("No valid edge")


class TestGraphPerturber:
    @pytest.fixture
    def test_graph(self):
        graph = nx.Graph()
        graph.add_node("A", id="A")
        graph.add_node("B", id="B")
        graph.add_node("C", id="C")
        graph.add_edge("A", "B", id="increase")
        graph.add_edge("B", "C", id="increase")
        return graph

    def test_init(self):
        mock_perturbation = MockPerturbation()
        perturber = perturbation.GraphPerturber(
            perturbations=[mock_perturbation],
            node_id_field="id",
            edge_id_field="id",
            p_prob=[1.0],
        )
        assert perturber.perturbations == [mock_perturbation]
        assert perturber.node_id_field == "id"
        assert perturber.edge_id_field == "id"
        assert perturber.perturbation_log == []
        assert perturber.total_edits == 0
        assert np.allclose(perturber.p_prob, [1.0])  # type: ignore

    def test_init_default_p_prob(self):
        mock_perturbation = MockPerturbation()
        perturber = perturbation.GraphPerturber(
            perturbations=[mock_perturbation],
            node_id_field="id",
            edge_id_field="id",
        )
        assert np.allclose(perturber.p_prob, [1.0])  # type: ignore

    def test_init_invalid_p_prob_length(self):
        mock_perturbation = MockPerturbation()
        with pytest.raises(ValueError):
            perturbation.GraphPerturber(
                perturbations=[mock_perturbation],
                node_id_field="id",
                edge_id_field="id",
                p_prob=[0.5, 0.5, 0.5],
            )

    def test_init_invalid_p_prob_sum(self):
        mock_perturbation = MockPerturbation()
        with pytest.raises(ValueError):
            perturbation.GraphPerturber(
                perturbations=[mock_perturbation],
                node_id_field="id",
                edge_id_field="id",
                p_prob=[0.5, 0.5],
            )

    def test_reset(self):
        mock_perturbation = MockPerturbation()
        perturber = perturbation.GraphPerturber(
            perturbations=[mock_perturbation],
            node_id_field="id",
            edge_id_field="id",
        )
        perturber.perturbation_log = [{"type": "edge_addition"}]
        perturber.total_edits = 5

        perturber.reset()

        assert perturber.perturbation_log == []
        assert perturber.total_edits == 0

    def test_perturb(self, test_graph: nx.Graph) -> None:
        mock_perturbation = MockPerturbation()
        perturber = perturbation.GraphPerturber(
            perturbations=[mock_perturbation],
            node_id_field="id",
            edge_id_field="id",
        )

        p_graph = perturber.perturb(test_graph, n_perturbations=2)

        # Check B -> C edge removed and A -> C added
        assert p_graph.has_edge("A", "C")
        assert not p_graph.has_edge("B", "C")

        # Check perturbation log is correctly updated
        assert perturber.perturbation_log == [
            {"type": "edge_deletion", "source": "B", "target": "C", "metadata": None},
            {"type": "edge_addition", "source": "A", "target": "C", "metadata": None},
        ]

        # Check total edits
        assert perturber.total_edits == 2

    def test_perturb_not_enough_valid_edges(self, test_graph: nx.Graph) -> None:
        mock_perturbation = MockPerturbation()
        perturber = perturbation.GraphPerturber(
            perturbations=[mock_perturbation],
            node_id_field="id",
            edge_id_field="id",
        )

        with pytest.raises(perturbation.NoValidEdgeError):
            perturber.perturb(test_graph, n_perturbations=3, max_retries=1)

    def test_perturb_duplicate(self, test_graph: nx.Graph) -> None:
        mock_perturbation = MockPerturbation()
        perturber = perturbation.GraphPerturber(
            perturbations=[mock_perturbation],
            node_id_field="id",
            edge_id_field="id",
        )
        # Append the first perturbation made by MockPerturbation to the log and
        # an error should be raised when trying to apply the same perturbation
        perturber.perturbation_log.append(
            {"type": "edge_deletion", "source": "B", "target": "C", "metadata": None},
        )
        with pytest.raises(perturbation.NoValidEdgeError):
            perturber.perturb(test_graph, n_perturbations=1, max_retries=1)
