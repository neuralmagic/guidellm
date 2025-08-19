"""
Unit tests for the auto_importer module.
"""

from unittest import mock

import pytest

from guidellm.utils import AutoImporterMixin


class MockHelper:
    """Helper class to create consistent mock objects for testing."""

    @staticmethod
    def create_mock_package(name: str, path: str):
        """Create a mock package with required attributes."""
        package = mock.MagicMock()
        package.__name__ = name
        package.__path__ = [path]
        return package

    @staticmethod
    def create_mock_module(name: str):
        """Create a mock module with required attributes."""
        module = mock.MagicMock()
        module.__name__ = name
        return module


class TestAutoImporterMixin:
    """Test suite for AutoImporterMixin functionality."""

    @pytest.mark.smoke
    def test_mixin_initialization(self):
        """Test that AutoImporterMixin initializes with correct default values."""
        assert AutoImporterMixin.auto_package is None
        assert AutoImporterMixin.auto_ignore_modules is None
        assert AutoImporterMixin.auto_imported_modules is None

    @pytest.mark.smoke
    def test_subclass_attributes(self):
        """Test that subclass can set auto_package attribute."""

        class TestClass(AutoImporterMixin):
            auto_package = "test.package"

        assert TestClass.auto_package == "test.package"
        assert TestClass.auto_ignore_modules is None
        assert TestClass.auto_imported_modules is None

    @pytest.mark.smoke
    def test_missing_package_raises_error(self):
        """Test that missing auto_package raises ValueError."""

        class TestClass(AutoImporterMixin):
            pass

        with pytest.raises(ValueError, match="auto_package.*must be set"):
            TestClass.auto_import_package_modules()

    @pytest.mark.smoke
    @mock.patch("importlib.import_module")
    @mock.patch("pkgutil.walk_packages")
    def test_single_package_import(self, mock_walk, mock_import):
        """Test importing modules from a single package."""

        class TestClass(AutoImporterMixin):
            auto_package = "test.package"

        # Setup mocks
        mock_package = MockHelper.create_mock_package("test.package", "test/package")
        mock_module1 = MockHelper.create_mock_module("test.package.module1")
        mock_module2 = MockHelper.create_mock_module("test.package.module2")

        mock_import.side_effect = lambda name: {
            "test.package": mock_package,
            "test.package.module1": mock_module1,
            "test.package.module2": mock_module2,
        }[name]

        mock_walk.return_value = [
            (None, "test.package.module1", False),
            (None, "test.package.module2", False),
        ]

        # Execute
        TestClass.auto_import_package_modules()

        # Verify
        assert TestClass.auto_imported_modules == [
            "test.package.module1",
            "test.package.module2",
        ]
        mock_import.assert_any_call("test.package")
        mock_import.assert_any_call("test.package.module1")
        mock_import.assert_any_call("test.package.module2")

    @pytest.mark.sanity
    @mock.patch("importlib.import_module")
    @mock.patch("pkgutil.walk_packages")
    def test_multiple_package_import(self, mock_walk, mock_import):
        """Test importing modules from multiple packages."""

        class TestClass(AutoImporterMixin):
            auto_package = ("test.package1", "test.package2")

        # Setup mocks
        packages = {
            "test.package1": MockHelper.create_mock_package(
                "test.package1", "test/package1"
            ),
            "test.package2": MockHelper.create_mock_package(
                "test.package2", "test/package2"
            ),
        }
        modules = {
            "test.package1.moduleA": MockHelper.create_mock_module(
                "test.package1.moduleA"
            ),
            "test.package2.moduleB": MockHelper.create_mock_module(
                "test.package2.moduleB"
            ),
        }

        mock_import.side_effect = lambda name: {**packages, **modules}[name]

        def walk_side_effect(path, prefix):
            if prefix == "test.package1.":
                return [(None, "test.package1.moduleA", False)]
            elif prefix == "test.package2.":
                return [(None, "test.package2.moduleB", False)]
            return []

        mock_walk.side_effect = walk_side_effect

        # Execute
        TestClass.auto_import_package_modules()

        # Verify
        assert TestClass.auto_imported_modules == [
            "test.package1.moduleA",
            "test.package2.moduleB",
        ]

    @pytest.mark.sanity
    @mock.patch("importlib.import_module")
    @mock.patch("pkgutil.walk_packages")
    def test_ignore_modules(self, mock_walk, mock_import):
        """Test that modules in auto_ignore_modules are skipped."""

        class TestClass(AutoImporterMixin):
            auto_package = "test.package"
            auto_ignore_modules = ("test.package.module1",)

        # Setup mocks
        mock_package = MockHelper.create_mock_package("test.package", "test/package")
        mock_module2 = MockHelper.create_mock_module("test.package.module2")

        mock_import.side_effect = lambda name: {
            "test.package": mock_package,
            "test.package.module2": mock_module2,
        }.get(name, mock.MagicMock())

        mock_walk.return_value = [
            (None, "test.package.module1", False),
            (None, "test.package.module2", False),
        ]

        # Execute
        TestClass.auto_import_package_modules()

        # Verify
        assert TestClass.auto_imported_modules == ["test.package.module2"]
        mock_import.assert_any_call("test.package")
        mock_import.assert_any_call("test.package.module2")
        # module1 should not be imported
        with pytest.raises(AssertionError):
            mock_import.assert_any_call("test.package.module1")

    @pytest.mark.sanity
    @mock.patch("importlib.import_module")
    @mock.patch("pkgutil.walk_packages")
    def test_skip_packages(self, mock_walk, mock_import):
        """Test that packages (is_pkg=True) are skipped."""

        class TestClass(AutoImporterMixin):
            auto_package = "test.package"

        # Setup mocks
        mock_package = MockHelper.create_mock_package("test.package", "test/package")
        mock_module = MockHelper.create_mock_module("test.package.module")

        mock_import.side_effect = lambda name: {
            "test.package": mock_package,
            "test.package.module": mock_module,
        }[name]

        mock_walk.return_value = [
            (None, "test.package.subpackage", True),
            (None, "test.package.module", False),
        ]

        # Execute
        TestClass.auto_import_package_modules()

        # Verify
        assert TestClass.auto_imported_modules == ["test.package.module"]
        mock_import.assert_any_call("test.package.module")
        # subpackage should not be imported
        with pytest.raises(AssertionError):
            mock_import.assert_any_call("test.package.subpackage")

    @pytest.mark.sanity
    @mock.patch("sys.modules", {"test.package.existing": mock.MagicMock()})
    @mock.patch("importlib.import_module")
    @mock.patch("pkgutil.walk_packages")
    def test_skip_already_imported_modules(self, mock_walk, mock_import):
        """Test that modules already in sys.modules are tracked but not re-imported."""

        class TestClass(AutoImporterMixin):
            auto_package = "test.package"

        # Setup mocks
        mock_package = MockHelper.create_mock_package("test.package", "test/package")
        mock_import.side_effect = lambda name: {
            "test.package": mock_package,
        }.get(name, mock.MagicMock())

        mock_walk.return_value = [
            (None, "test.package.existing", False),
        ]

        # Execute
        TestClass.auto_import_package_modules()

        # Verify
        assert TestClass.auto_imported_modules == ["test.package.existing"]
        mock_import.assert_called_once_with("test.package")
        with pytest.raises(AssertionError):
            mock_import.assert_any_call("test.package.existing")

    @pytest.mark.sanity
    @mock.patch("importlib.import_module")
    @mock.patch("pkgutil.walk_packages")
    def test_prevent_duplicate_module_imports(self, mock_walk, mock_import):
        """Test that modules already in auto_imported_modules are not re-imported."""

        class TestClass(AutoImporterMixin):
            auto_package = "test.package"

        # Setup mocks
        mock_package = MockHelper.create_mock_package("test.package", "test/package")
        mock_module = MockHelper.create_mock_module("test.package.module")

        mock_import.side_effect = lambda name: {
            "test.package": mock_package,
            "test.package.module": mock_module,
        }[name]

        mock_walk.return_value = [
            (None, "test.package.module", False),
            (None, "test.package.module", False),
        ]

        # Execute
        TestClass.auto_import_package_modules()

        # Verify
        assert TestClass.auto_imported_modules == ["test.package.module"]
        assert mock_import.call_count == 2  # Package + module (not duplicate)
