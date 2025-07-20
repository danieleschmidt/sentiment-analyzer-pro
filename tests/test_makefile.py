"""Tests for Makefile functionality and development workflow."""

import subprocess
import os
import pytest
from pathlib import Path


class TestMakefileWorkflow:
    """Test the Makefile provides a reliable development workflow."""
    
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Ensure we're in the project root for Makefile tests."""
        self.project_root = Path(__file__).parent.parent
        self.original_cwd = os.getcwd()
        os.chdir(self.project_root)
        yield
        os.chdir(self.original_cwd)
    
    def run_make_command(self, target: str) -> subprocess.CompletedProcess:
        """Helper to run make commands and capture output."""
        return subprocess.run(
            ["make", target],
            capture_output=True,
            text=True,
            timeout=30
        )
    
    def test_makefile_exists(self):
        """Test that Makefile exists in project root."""
        assert Path("Makefile").exists(), "Makefile should exist in project root"
    
    def test_make_help_command(self):
        """Test that make help provides usage information."""
        result = self.run_make_command("help")
        assert result.returncode == 0, f"make help failed: {result.stderr}"
        assert "Sentiment Analyzer Pro" in result.stdout
        assert "setup" in result.stdout
        assert "test" in result.stdout
        
    def test_make_version_command(self):
        """Test that make version returns package version."""
        result = self.run_make_command("version")
        assert result.returncode == 0, f"make version failed: {result.stderr}"
        # Should return semantic version format
        version_output = result.stdout.strip()
        assert len(version_output) > 0, "Version command should return version string"
        # Basic semantic version pattern check
        assert "." in version_output, "Version should contain dots"
    
    def test_make_debug_env_command(self):
        """Test that debug-env provides environment information."""
        result = self.run_make_command("debug-env")
        assert result.returncode == 0, f"make debug-env failed: {result.stderr}"
        assert "Python:" in result.stdout
        assert "Working Directory:" in result.stdout
        
    def test_make_clean_command(self):
        """Test that clean command removes build artifacts."""
        # Create some dummy build artifacts
        test_dirs = ["build", "dist", "test.egg-info"]
        test_files = [".coverage", "security-report.json"]
        
        for dir_name in test_dirs:
            os.makedirs(dir_name, exist_ok=True)
            
        for file_name in test_files:
            Path(file_name).touch()
            
        # Run clean command
        result = self.run_make_command("clean")
        assert result.returncode == 0, f"make clean failed: {result.stderr}"
        
        # Verify artifacts are removed
        for dir_name in test_dirs:
            assert not Path(dir_name).exists(), f"{dir_name} should be removed by clean"
            
        for file_name in test_files:
            assert not Path(file_name).exists(), f"{file_name} should be removed by clean"
    
    def test_make_install_dry_run(self):
        """Test that install command structure is valid (dry run style check)."""
        # We won't actually install to avoid modifying the environment
        # Instead, verify the Makefile has the expected targets
        with open("Makefile", "r") as f:
            makefile_content = f.read()
            
        required_targets = [
            "install:", "install-dev:", "install-ml:", "install-web:",
            "test:", "lint:", "format:", "check:", "security:",
            "dev:", "serve:", "build:", "clean:"
        ]
        
        for target in required_targets:
            assert target in makefile_content, f"Target {target} missing from Makefile"
    
    def test_makefile_syntax_validation(self):
        """Test that Makefile has valid syntax by parsing available targets."""
        result = subprocess.run(
            ["make", "-n", "help"],  # -n for dry run
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        assert result.returncode == 0, f"Makefile syntax error: {result.stderr}"
    
    def test_color_output_disabled_in_tests(self):
        """Test that Makefile commands work in non-interactive environments."""
        # Set environment to disable colors
        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        
        result = subprocess.run(
            ["make", "help"],
            capture_output=True,
            text=True,
            env=env,
            cwd=self.project_root
        )
        assert result.returncode == 0, "Makefile should work without color output"


class TestMakefileDocumentation:
    """Test that Makefile is properly documented."""
    
    def test_makefile_has_help_comments(self):
        """Test that all major targets have help comments."""
        with open("Makefile", "r") as f:
            content = f.read()
            
        # Check that help target exists and has proper format
        assert "help:" in content
        assert "## Show this help message" in content
        
        # Check for presence of help comments on major targets
        help_patterns = [
            "setup.*##", "install.*##", "test.*##", 
            "lint.*##", "clean.*##", "build.*##"
        ]
        
        import re
        for pattern in help_patterns:
            assert re.search(pattern, content), f"Missing help comment matching: {pattern}"
    
    def test_makefile_has_proper_phony_declarations(self):
        """Test that non-file targets are declared as .PHONY."""
        with open("Makefile", "r") as f:
            content = f.read()
            
        assert ".PHONY:" in content, "Makefile should declare phony targets"
        
        # Key targets should be declared as phony
        phony_targets = ["help", "setup", "install", "test", "clean", "build"]
        for target in phony_targets:
            assert target in content, f"Target {target} should be declared as phony"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])