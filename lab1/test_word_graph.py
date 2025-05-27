import pytest
from word_graph import WordGraph, TextProcessor

@pytest.fixture
def graph():
    g = WordGraph()
    words = TextProcessor.process_text("To explore strange new worlds,To seek out new life and new civilizations")
    g.build_graph(words)
    return g

def test_query_bridge_words_exists(graph):
    assert graph.query_bridge_words("strange", "worlds") == ["new"]

def test_query_bridge_words_none(graph):
    assert graph.query_bridge_words("civilizations", "and") == []

def test_query_bridge_words_case_insensitive(graph):
    assert graph.query_bridge_words("Strange", "woRlds") == ["new"]

def test_query_bridge_words_invalid_word(graph):
    assert graph.query_bridge_words("at", "to") == []

