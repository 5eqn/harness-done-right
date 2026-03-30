"""
Test the example workflow to ensure it works correctly
"""
import os
import tempfile
import hdr
from hdr import save_config

def test_example_workflow():
    """Test that the example pattern works correctly"""
    # Create temporary directory for test config
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override HDR paths for test
        original_hdr_dir = hdr.HDR_DIR
        original_config_file = hdr.CONFIG_FILE
        original_log_file = hdr.LOG_FILE

        hdr.HDR_DIR = tmpdir
        hdr.CONFIG_FILE = os.path.join(tmpdir, "config.json")
        hdr.LOG_FILE = os.path.join(tmpdir, "llm_logs.jsonl")

        # Enable mock mode - FOR TEST USE ONLY
        # Mock mode is strictly limited to pytest testing, never use in production
        save_config({"openrouter_model": "mock"})

        # Import example modules (must import after config is set)
        from example.task import IntroductionSection, UsageSection, Documentation

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
        hdr.LOG_FILE = original_log_file

    print("✅ Example workflow test passed!")

if __name__ == "__main__":
    test_example_workflow()
