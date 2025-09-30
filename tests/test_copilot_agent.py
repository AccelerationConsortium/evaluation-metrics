"""Test to verify GitHub Copilot coding agent functionality."""


def test_copilot_agent_running():
    """Verify that the GitHub Copilot coding agent can run successfully."""
    assert True, "Copilot coding agent is running!"


def test_basic_arithmetic():
    """Simple test to demonstrate agent can write basic test code."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6
    assert 10 / 2 == 5


def test_string_operations():
    """Test basic string operations."""
    greeting = "Hello, Copilot!"
    assert greeting.startswith("Hello")
    assert "Copilot" in greeting
    assert len(greeting) > 0
