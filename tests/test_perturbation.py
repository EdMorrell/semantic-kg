from __future__ import annotations

import pytest
import numpy as np
import networkx as nx
from typing import Optional

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

        graph.add_node("A", id="A")
        graph.add_node("B", id="B")
        graph.add_node("C", id="C")

        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        return graph

    def test_create(self, test_graph: nx.Graph) -> None:
        perturber = perturbation.EdgeAdditionPerturbation(node_type_id="id")
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
        valid_edge_names = ["A_B", "B_C"]

        perturber = perturbation.EdgeAdditionPerturbation(
            node_type_id="id", valid_edge_types=valid_edge_names
        )
        with pytest.raises(perturbation.NoValidEdgeError):
            _ = perturber.create(test_graph)

    def test_create_valid_edges_edge_attribute_mapper(
        self, test_graph: nx.Graph
    ) -> None:
        """Only allow edges that already exist which should raise an error"""

        # Mock mapper simply adds the source and target node id types as
        # attributes
        class MockAttributeMapper:
            def get_attributes(
                self,
                src_node: dict,
                target_node: dict,
                edge_value: Optional[str] = None,
            ) -> dict:
                src_id = src_node["id"]
                target_id = target_node["id"]

                return {"src_node_id": src_id, "target_node_id": target_id}

        perturber = perturbation.EdgeAdditionPerturbation(
            node_type_id="id",
            edge_attribute_mapper=MockAttributeMapper(),
        )

        p_graph = perturber.create(test_graph)

        # Check an edges exists between A and C
        assert "C" in list(p_graph.neighbors("A"))

        # Check the edge attributes
        edge_data = p_graph.get_edge_data("A", "C")

        assert (
            edge_data["src_node_id"] == "A" and edge_data["target_node_id"] == "C"
        ) or (edge_data["src_node_id"] == "C" and edge_data["target_node_id"] == "A")

    def test_reset(self, test_graph: nx.Graph) -> None:
        perturber = perturbation.EdgeAdditionPerturbation(node_type_id="id")
        _ = perturber.create(test_graph)

        perturber.reset()
        assert perturber.edit_count == 0
        assert perturber.perturbation_log == []

    def test_create_directed(self, test_di_graph: nx.DiGraph) -> None:
        perturber = perturbation.EdgeAdditionPerturbation(node_type_id="id")
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
        perturber = perturbation.EdgeDeletionPerturbation()
        p_graph = perturber.create(test_graph)

        assert p_graph.number_of_edges() == test_graph.number_of_edges() - 1
        assert "B" not in list(p_graph.neighbors("A"))
        assert perturber.edit_count == 1

    def test_create_perturbation_log(self, test_graph: nx.Graph) -> None:
        perturber = perturbation.EdgeDeletionPerturbation()
        _ = perturber.create(test_graph)

        assert len(perturber.perturbation_log) == 1
        assert perturber.perturbation_log[0]["type"] == "edge_deletion"
        assert perturber.perturbation_log[0]["source"] in ["A", "B"]
        assert perturber.perturbation_log[0]["target"] in ["A", "B"]

    def test_invalid_create(self, test_graph: nx.Graph) -> None:
        perturber = perturbation.EdgeDeletionPerturbation()
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

        graph.add_node("A", id="A", node_type="allowed")
        graph.add_node("B", id="B", node_type="allowed")
        graph.add_node("C", id="C", node_type="allowed")

        graph.add_edge("A", "B", effect="increase")
        graph.add_edge("B", "C", effect="decrease")

        return graph

    @pytest.fixture(scope="class")
    def test_attribute_mapper(
        self, test_graph: nx.Graph
    ) -> perturbation.EdgeAttributeMapper:
        class MockAttributeMapper:
            def get_attributes(
                self,
                src_node: dict,
                target_node: dict,
                edge_value: Optional[str] = None,
            ) -> dict:
                return {"effect": edge_value}

        return MockAttributeMapper()

    def test_create(
        self,
        test_graph: nx.Graph,
        test_attribute_mapper: perturbation.EdgeAttributeMapper,
    ) -> None:
        perturber = perturbation.EdgeReplacementPerturbation(
            node_type_id="node_type",
            edge_name_id="effect",
            replace_map={
                "allowed_allowed": ["increase", "decrease"],
            },
            edge_attribute_mapper=test_attribute_mapper,
        )

        p_graph = perturber.create(test_graph)

        assert p_graph.number_of_edges() == test_graph.number_of_edges()
        # Find the replaced edge
        replaced_edge = None
        for u, v, data in test_graph.edges(data=True):
            if data["effect"] != p_graph.get_edge_data(u, v)["effect"]:
                replaced_edge = (u, v)

        # Checks a replaced edge is found
        assert replaced_edge is not None
        # Checks the edge `effect` attribute has been changed
        assert (
            p_graph.get_edge_data(*replaced_edge)["effect"]
            != test_graph.get_edge_data(*replaced_edge)["effect"]
        )

        assert perturber.edit_count == 1

        # Check log is correctly updated
        assert perturber.perturbation_log[-1]["type"] == "edge_replacement"
        assert perturber.perturbation_log[-1]["source"] in list(replaced_edge)
        assert perturber.perturbation_log[-1]["target"] in list(replaced_edge)
        assert perturber.perturbation_log[-1]["metadata"] == {
            "effect": p_graph.get_edge_data(*replaced_edge)["effect"]
        }

    def test_create_invalid(
        self,
        test_graph: nx.Graph,
        test_attribute_mapper: perturbation.EdgeAttributeMapper,
    ) -> None:
        modified_graph = test_graph.copy()

        # Update both node attributes so the type is `disallowed`
        nx.set_node_attributes(modified_graph, "disallowed", "node_type")

        perturber = perturbation.EdgeReplacementPerturbation(
            node_type_id="node_type",
            edge_name_id="effect",
            replace_map={
                "allowed_allowed": ["increase", "decrease"],
            },
            edge_attribute_mapper=test_attribute_mapper,
        )

        with pytest.raises(perturbation.NoValidEdgeError):
            _ = perturber.create(modified_graph)

    def test_create_directed(
        self,
        test_di_graph: nx.DiGraph,
        test_attribute_mapper: perturbation.EdgeAttributeMapper,
    ) -> None:
        perturber = perturbation.EdgeReplacementPerturbation(
            node_type_id="node_type",
            edge_name_id="effect",
            replace_map={
                "allowed_allowed": ["increase", "decrease"],
            },
            edge_attribute_mapper=test_attribute_mapper,
        )
        p_graph = perturber.create(test_di_graph)
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
