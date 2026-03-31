"""
Test the example workflow to ensure it works correctly.

IMPORTANT: Mock mode is ONLY for pytest testing purposes!
Do NOT use mock mode in any real execution environment.
"""
import os
import tempfile
import hdr
from hdr import save_config

# Enable mock mode for tests
# WARNING: Mock mode should ONLY be used in pytest unit tests
hdr.set_mock_mode(True)

def test_example_workflow():
    """Test that the example pattern works correctly"""
    # Create temporary directory for test config
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override HDR paths for test
        original_hdr_dir = hdr.HDR_DIR
        original_config_file = hdr.CONFIG_FILE

        hdr.HDR_DIR = tmpdir
        hdr.CONFIG_FILE = os.path.join(tmpdir, "config.json")

        save_config({})

        # Import example modules (must import after config is set)
        from example.introduction_writing.task import IntroductionSection, UsageSection, Documentation

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

        # Restore original paths
        hdr.HDR_DIR = original_hdr_dir
        hdr.CONFIG_FILE = original_config_file

    print("✅ Example workflow test passed!")

if __name__ == "__main__":
    test_example_workflow()
