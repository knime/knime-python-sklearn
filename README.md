
# Sklearn Extension for KNIME

This repository is the home of the [Sklearn Extension]() for [KNIME Analytics Platform](https://www.knime.com/knime-analytics-platform). The extension provides a set of nodes for regression and classification.

The extension is mainly based on the [scikit-learn](https://scikit-learn.org/stable/) library.

[Here are some examples of workflows]() that use the extension.


## How to Contribute

### Package Organization

* `knime_extension`: This folder contains all files of the KNIME Sklearn extension such as the source code of each node. The folder itself is structured as suggested in the [Python Best Practices]()file.
* `docs`: Additional material to get you started with the development of this extension such as development setup instructions or best practices.
* `tests`: Test data and workflows used to test the node functionality.
* `config.yml`: Example `config.yml` file that should point to the `knime_extension` folder on your local hard drive during local development and debugging as described [here](https://docs.knime.com/latest/pure_python_node_extensions_guide/index.html#tutorial-writing-first-py-node).


### Contribute Guidelines

TODO


## License
The repository is released under the [MIT License](https://opensource.org/licenses/MIT).

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)