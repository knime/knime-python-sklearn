# KNIME Nodes for Scikit-Learn (sklearn) Algorithms

[![Jenkins](https://jenkins.knime.com/buildStatus/icon?job=knime-python-sklearn%2Fmaster)](https://jenkins.knime.com/job/knime-python-sklearn/job/master)

This repository is maintained by the [KNIME Team Rakete](mailto:team-rakete@knime.com).

This repository contains the source code of the [KNIME Nodes for Scikit-Learn (sklearn) Algorithms](https://hub.knime.com/knime/extensions/org.knime.python.features.sklearn/latest) for [KNIME Analytics Platform](https://www.knime.com/knime-analytics-platform).
The KNIME nodes in this extension make algorithms of the [scikit-learn](https://scikit-learn.org/stable/) library available in KNIME.

## Development notes

Please see the [Python extension development documentation at `docs.knime.com`](https://docs.knime.com/latest/pure_python_node_extensions_guide/index.html#introduction)
for more information on how to build pure-Python nodes for KNIME.

If you want to add nodes for more scikit-learn algorithms to this repository, clone it locally, 
create a `config.yml` (e.g. using the template in this repo) file pointing to this folder and add 
an entry in the `knime.ini` referencing this `config.yml` file as described 
[here](https://docs.knime.com/latest/pure_python_node_extensions_guide/index.html#registering-python-extensions-during-development).

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Join the Community

* [KNIME Forum](https://forum.knime.com/)

## License

The repository is released under the [GPL v3 License](https://www.gnu.org/licenses/gpl-3.0.html). Please refer to `LICENSE.txt`.
