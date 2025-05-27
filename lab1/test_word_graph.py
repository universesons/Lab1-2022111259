import pytest
from word_graph import WordGraph, TextProcessor

@pytest.fixture
def graph():
    g = WordGraph()
    words = TextProcessor.process_text("To explore strange new worlds,To seek out new life and new civilizations")
    g.build_graph(words)
    return g

# def test_query_bridge_words_exists(graph):
#     assert graph.query_bridge_words("strange", "worlds") == ["new"]
#
# def test_query_bridge_words_none(graph):
#     assert graph.query_bridge_words("civilizations", "and") == []
#
# def test_query_bridge_words_case_insensitive(graph):
#     assert graph.query_bridge_words("Strange", "woRlds") == ["new"]
#
# def test_query_bridge_words_invalid_word(graph):
#     assert graph.query_bridge_words("at", "to") == []
def test_word1_not_in_nodes(graph):
    paths, distance = graph.calc_shortest_path("a", "b")
    assert paths == []
    assert distance == 0

def test_target_not_in_nodes(graph):
    paths, distance = graph.calc_shortest_path("and", "b")
    assert paths == []
    assert distance == 0

def test_target_unreachable(graph):
    paths, distance = graph.calc_shortest_path("civilization", "new")
    assert paths == []
    assert distance == 0

def test_target_reachable(graph):
    paths, distance = graph.calc_shortest_path("and", "new")
    assert paths == [("and","new")]
    assert distance == 1