import toml
from loguru import logger

import guidellm


def construct_project_name():
    project_names = {
        "release": "guidellm",
        "nightly": "guidellm_nightly",
        "dev": "guidellm_dev",
    }
    return project_names.get(guidellm.build_type, "guidellm")


def update_pyproject_toml(project_name):
    try:
        with open("pyproject.toml", "r") as file:
            data = toml.load(file)

        data["project"]["name"] = project_name

        with open("pyproject.toml", "w") as file:
            toml.dump(data, file)

        logger.info(f"Updated project name to: {project_name}")
    except (FileNotFoundError, toml.TomlDecodeError) as e:
        logger.error(f"Error reading or writing pyproject.toml: {e}")
        raise


def main():
    try:
        project_name = construct_project_name()
        update_pyproject_toml(project_name)
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
