"""
Test the example workflow to ensure it works correctly.

IMPORTANT: Mock mode is ONLY for pytest testing purposes!
Do NOT use mock mode in any real execution environment.
"""
import tempfile
import hdr
from example.introduction_writing.task import IntroductionSection, UsageSection, Documentation

# Enable mock mode for tests
# WARNING: Mock mode should ONLY be used in pytest unit tests
hdr.set_mock_mode(True)

def test_example_workflow():
    """Test that the example pattern works correctly"""
    # Test constructing subtasks
    intro = IntroductionSection(
        content="Test introduction content"
    )
    assert isinstance(intro, IntroductionSection)
    assert intro.content == "Test introduction content"

    usage = UsageSection(
        content="Test usage content",
        code_examples=["test1", "test2"]
    )
    assert isinstance(usage, UsageSection)
    assert usage.content == "Test usage content"
    assert len(usage.code_examples) == 2

    # Test constructing final task
    doc = Documentation(
        title="Test Documentation",
        introduction=intro,
        usage=usage
    )
    assert isinstance(doc, Documentation)
    assert doc.title == "Test Documentation"
    assert doc.introduction == intro
    assert doc.usage == usage

    print("✅ Example workflow test passed!")

if __name__ == "__main__":
    test_example_workflow()
