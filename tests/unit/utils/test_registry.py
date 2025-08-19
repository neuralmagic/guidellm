"""
Unit tests for the registry module.
"""

from unittest import mock

import pytest

from guidellm.utils.registry import RegistryMixin


class TestBasicRegistration:
    """Test suite for basic registry functionality."""

    @pytest.mark.smoke
    def test_registry_initialization(self):
        """Test that RegistryMixin initializes with correct defaults."""

        class TestRegistryClass(RegistryMixin):
            pass

        assert TestRegistryClass.registry is None
        assert TestRegistryClass.registry_auto_discovery is False
        assert TestRegistryClass.registry_populated is False

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("register_name", "expected_key"),
        [
            ("custom_name", "custom_name"),
            ("CamelCase", "camelcase"),
            ("UPPERCASE", "uppercase"),
            ("snake_case", "snake_case"),
        ],
    )
    def test_register_with_name(self, register_name, expected_key):
        """Test registering objects with explicit names."""

        class TestRegistryClass(RegistryMixin):
            pass

        @TestRegistryClass.register(register_name)
        class TestClass:
            pass

        assert TestRegistryClass.registry is not None
        assert expected_key in TestRegistryClass.registry
        assert TestRegistryClass.registry[expected_key] is TestClass

    @pytest.mark.smoke
    def test_register_without_name(self):
        """Test registering objects without explicit names."""

        class TestRegistryClass(RegistryMixin):
            pass

        @TestRegistryClass.register()
        class TestClass:
            pass

        assert TestRegistryClass.registry is not None
        assert "testclass" in TestRegistryClass.registry
        assert TestRegistryClass.registry["testclass"] is TestClass

    @pytest.mark.smoke
    def test_register_decorator_direct(self):
        """Test direct usage of register_decorator."""

        class TestRegistryClass(RegistryMixin):
            pass

        @TestRegistryClass.register_decorator
        class TestClass:
            pass

        assert TestRegistryClass.registry is not None
        assert "testclass" in TestRegistryClass.registry
        assert TestRegistryClass.registry["testclass"] is TestClass

    @pytest.mark.smoke
    def test_register_multiple_names(self):
        """Test registering an object with multiple names."""

        class TestRegistryClass(RegistryMixin):
            pass

        @TestRegistryClass.register(["name1", "name2", "Name3"])
        class TestClass:
            pass

        assert TestRegistryClass.registry is not None
        assert "name1" in TestRegistryClass.registry
        assert "name2" in TestRegistryClass.registry
        assert "name3" in TestRegistryClass.registry
        assert all(
            TestRegistryClass.registry[key] is TestClass
            for key in ["name1", "name2", "name3"]
        )

    @pytest.mark.smoke
    def test_registered_objects(self):
        """Test retrieving all registered objects."""

        class TestRegistryClass(RegistryMixin):
            pass

        @TestRegistryClass.register()
        class TestClass1:
            pass

        @TestRegistryClass.register("custom_name")
        class TestClass2:
            pass

        registered = TestRegistryClass.registered_objects()
        assert isinstance(registered, tuple)
        assert len(registered) == 2
        assert TestClass1 in registered
        assert TestClass2 in registered


class TestRegistrationValidation:
    """Test suite for registration validation and error handling."""

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "invalid_name", [123, 42.5, True, {"key": "value"}, object()]
    )
    def test_register_invalid_name_type(self, invalid_name):
        """Test that invalid name types raise ValueError."""

        class TestRegistryClass(RegistryMixin):
            pass

        with pytest.raises(ValueError, match="name must be a string, list of strings"):
            TestRegistryClass.register(invalid_name)

    @pytest.mark.sanity
    def test_register_decorator_invalid_object(self):
        """Test that register_decorator validates object has __name__ attribute."""

        class TestRegistryClass(RegistryMixin):
            pass

        with pytest.raises(AttributeError):
            TestRegistryClass.register_decorator("not_a_class")

    @pytest.mark.sanity
    @pytest.mark.parametrize("invalid_name", [123, 42.5, True, {"key": "value"}])
    def test_register_decorator_invalid_name_type(self, invalid_name):
        """Test that invalid name types in register_decorator raise ValueError."""

        class TestRegistryClass(RegistryMixin):
            pass

        class TestClass:
            pass

        with pytest.raises(
            ValueError, match="name must be a string or an iterable of strings"
        ):
            TestRegistryClass.register_decorator(TestClass, name=invalid_name)

    @pytest.mark.sanity
    def test_register_decorator_invalid_list_element(self):
        """Test that invalid elements in name list raise ValueError."""

        class TestRegistryClass(RegistryMixin):
            pass

        class TestClass:
            pass

        with pytest.raises(
            ValueError, match="name must be a string or a list of strings"
        ):
            TestRegistryClass.register_decorator(TestClass, name=["valid", 123])

    @pytest.mark.sanity
    def test_register_duplicate_name(self):
        """Test that duplicate names raise ValueError."""

        class TestRegistryClass(RegistryMixin):
            pass

        @TestRegistryClass.register("test_name")
        class TestClass1:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @TestRegistryClass.register("test_name")
            class TestClass2:
                pass

    @pytest.mark.sanity
    def test_registered_objects_empty_registry(self):
        """Test that registered_objects raises error when no objects registered."""

        class TestRegistryClass(RegistryMixin):
            pass

        with pytest.raises(
            ValueError, match="must be called after registering objects"
        ):
            TestRegistryClass.registered_objects()


class TestRegistryIsolation:
    """Test suite for registry isolation between different classes."""

    @pytest.mark.regression
    def test_multiple_registries_isolation(self):
        """Test that different registry classes maintain separate registries."""

        class Registry1(RegistryMixin):
            pass

        class Registry2(RegistryMixin):
            pass

        @Registry1.register()
        class TestClass1:
            pass

        @Registry2.register()
        class TestClass2:
            pass

        assert Registry1.registry is not None
        assert Registry2.registry is not None
        assert Registry1.registry != Registry2.registry
        assert "testclass1" in Registry1.registry
        assert "testclass2" in Registry2.registry
        assert "testclass1" not in Registry2.registry
        assert "testclass2" not in Registry1.registry

    @pytest.mark.regression
    def test_inheritance_registry_sharing(self):
        """Test that inherited registry classes share the same registry."""

        class BaseRegistry(RegistryMixin):
            pass

        class ChildRegistry(BaseRegistry):
            pass

        @BaseRegistry.register()
        class BaseClass:
            pass

        @ChildRegistry.register()
        class ChildClass:
            pass

        # Child classes share the same registry as their parent
        assert BaseRegistry.registry is ChildRegistry.registry

        # Both classes can see all registered objects
        base_objects = BaseRegistry.registered_objects()
        child_objects = ChildRegistry.registered_objects()

        assert len(base_objects) == 2
        assert len(child_objects) == 2
        assert base_objects == child_objects
        assert BaseClass in base_objects
        assert ChildClass in base_objects


class TestAutoDiscovery:
    """Test suite for auto-discovery functionality."""

    @pytest.mark.smoke
    def test_auto_discovery_initialization(self):
        """Test initialization of auto-discovery enabled registry."""

        class TestAutoRegistry(RegistryMixin):
            registry_auto_discovery = True
            auto_package = "test_package.modules"

        assert TestAutoRegistry.registry is None
        assert TestAutoRegistry.registry_populated is False
        assert TestAutoRegistry.auto_package == "test_package.modules"
        assert TestAutoRegistry.registry_auto_discovery is True

    @pytest.mark.smoke
    def test_auto_populate_registry(self):
        """Test auto population mechanism."""

        class TestAutoRegistry(RegistryMixin):
            registry_auto_discovery = True
            auto_package = "test_package.modules"

        with mock.patch.object(
            TestAutoRegistry, "auto_import_package_modules"
        ) as mock_import:
            result = TestAutoRegistry.auto_populate_registry()
            assert result is True
            mock_import.assert_called_once()
            assert TestAutoRegistry.registry_populated is True

            result = TestAutoRegistry.auto_populate_registry()
            assert result is False
            mock_import.assert_called_once()

    @pytest.mark.sanity
    def test_auto_populate_registry_disabled(self):
        """Test that auto population fails when disabled."""

        class TestDisabledAutoRegistry(RegistryMixin):
            auto_package = "test_package.modules"

        with pytest.raises(ValueError, match="registry_auto_discovery is set to False"):
            TestDisabledAutoRegistry.auto_populate_registry()

    @pytest.mark.sanity
    def test_auto_registered_objects(self):
        """Test automatic population during registered_objects call."""

        class TestAutoRegistry(RegistryMixin):
            registry_auto_discovery = True
            auto_package = "test_package.modules"

        with mock.patch.object(
            TestAutoRegistry, "auto_populate_registry"
        ) as mock_populate:
            TestAutoRegistry.registry = {"class1": "obj1", "class2": "obj2"}
            objects = TestAutoRegistry.registered_objects()
            mock_populate.assert_called_once()
            assert objects == ("obj1", "obj2")


class TestAutoDiscoveryIntegration:
    """Test suite for comprehensive auto-discovery integration scenarios."""

    @pytest.mark.regression
    def test_auto_registry_integration(self):
        """Test complete auto-discovery workflow with mocked imports."""

        class TestAutoRegistry(RegistryMixin):
            registry_auto_discovery = True
            auto_package = "test_package.modules"

        with (
            mock.patch("pkgutil.walk_packages") as mock_walk,
            mock.patch("importlib.import_module") as mock_import,
        ):
            mock_package = mock.MagicMock()
            mock_package.__path__ = ["test_package/modules"]
            mock_package.__name__ = "test_package.modules"

            def import_module(name: str):
                if name == "test_package.modules":
                    return mock_package
                elif name == "test_package.modules.module1":
                    module = mock.MagicMock()
                    module.__name__ = "test_package.modules.module1"

                    class Module1Class:
                        pass

                    TestAutoRegistry.register_decorator(Module1Class, "Module1Class")
                    return module
                else:
                    raise ImportError(f"No module named {name}")

            def walk_packages(package_path, package_name):
                if package_name == "test_package.modules.":
                    return [(None, "test_package.modules.module1", False)]
                else:
                    raise ValueError(f"Unknown package: {package_name}")

            mock_walk.side_effect = walk_packages
            mock_import.side_effect = import_module

            objects = TestAutoRegistry.registered_objects()
            assert len(objects) == 1
            assert TestAutoRegistry.registry_populated is True
            assert TestAutoRegistry.registry is not None
            assert "module1class" in TestAutoRegistry.registry

    @pytest.mark.regression
    def test_auto_registry_multiple_packages(self):
        """Test auto-discovery with multiple packages."""

        class TestMultiPackageRegistry(RegistryMixin):
            registry_auto_discovery = True
            auto_package = ("package1", "package2")

        with mock.patch.object(
            TestMultiPackageRegistry, "auto_import_package_modules"
        ) as mock_import:
            TestMultiPackageRegistry.registry = {}
            TestMultiPackageRegistry.registered_objects()
            mock_import.assert_called_once()
            assert TestMultiPackageRegistry.registry_populated is True

    @pytest.mark.regression
    def test_auto_registry_import_error(self):
        """Test handling of import errors during auto-discovery."""

        class TestErrorRegistry(RegistryMixin):
            registry_auto_discovery = True
            auto_package = "nonexistent.package"

        with mock.patch.object(
            TestErrorRegistry,
            "auto_import_package_modules",
            side_effect=ValueError("auto_package must be set"),
        ) as mock_import:
            with pytest.raises(ValueError, match="auto_package must be set"):
                TestErrorRegistry.auto_populate_registry()
            mock_import.assert_called_once()
