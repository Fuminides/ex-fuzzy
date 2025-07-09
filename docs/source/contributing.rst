============
Contributing
============

We welcome contributions to Ex-Fuzzy! This guide will help you get started with contributing code, documentation, or examples.

Ways to Contribute
==================

There are many ways to contribute to Ex-Fuzzy:

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 🐛 Report Bugs
        :link: #reporting-bugs
        :link-type: ref

        Found a bug? Help us fix it by reporting detailed information.

    .. grid-item-card:: 💡 Suggest Features
        :link: #suggesting-features  
        :link-type: ref

        Have an idea for a new feature? We'd love to hear about it!

    .. grid-item-card:: 📝 Improve Documentation
        :link: #documentation
        :link-type: ref

        Help make our documentation clearer and more comprehensive.

    .. grid-item-card:: 💻 Contribute Code
        :link: #code-contributions
        :link-type: ref

        Fix bugs, implement features, or optimize performance.

Getting Started
===============

Development Setup
-----------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

       git clone https://github.com/your-username/ex-fuzzy.git
       cd ex-fuzzy

3. **Create a virtual environment**:

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install in development mode**:

   .. code-block:: bash

       pip install -e ".[dev,test,docs]"

5. **Install pre-commit hooks**:

   .. code-block:: bash

       pre-commit install

Code Style and Standards
=========================

We follow established Python coding standards:

- **PEP 8** for code style
- **Type hints** for all public functions
- **Comprehensive docstrings** in NumPy format
- **Unit tests** for all new functionality

Code Formatting
---------------

We use automated tools to maintain code quality:

.. code-block:: bash

    # Format code with black
    black ex_fuzzy/

    # Sort imports with isort
    isort ex_fuzzy/

    # Lint with flake8
    flake8 ex_fuzzy/

    # Type checking with mypy
    mypy ex_fuzzy/

Testing
-------

Run the test suite before submitting changes:

.. code-block:: bash

    # Run all tests
    pytest

    # Run with coverage
    pytest --cov=ex_fuzzy --cov-report=html

    # Run specific test file
    pytest tests/test_fuzzy_sets.py

Contribution Workflow
=====================

1. **Create a new branch** for your feature:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. **Make your changes** following our coding standards

3. **Add tests** for any new functionality

4. **Update documentation** if needed

5. **Run the test suite** to ensure everything works

6. **Commit your changes** with a clear message:

   .. code-block:: bash

       git commit -m "Add feature: brief description of what you added"

7. **Push to your fork**:

   .. code-block:: bash

       git push origin feature/your-feature-name

8. **Create a Pull Request** on GitHub

Pull Request Guidelines
=======================

When submitting a pull request:

**Title and Description**
  - Use a clear, descriptive title
  - Explain what your changes do and why
  - Reference any related issues

**Code Quality**
  - Ensure all tests pass
  - Follow our coding standards
  - Include appropriate documentation

**Review Process**
  - Be responsive to feedback
  - Make requested changes promptly
  - Update your branch if needed

.. _reporting-bugs:

Reporting Bugs
==============

Before reporting a bug:

1. **Check existing issues** to avoid duplicates
2. **Update to the latest version** to see if it's already fixed
3. **Create a minimal example** that reproduces the bug

When reporting, include:

- Ex-Fuzzy version
- Python version and operating system
- Complete error traceback
- Minimal code example
- Expected vs actual behavior

Use our bug report template:

.. code-block:: markdown

    **Bug Description**
    A clear description of what the bug is.

    **To Reproduce**
    ```python
    # Minimal code example
    ```

    **Expected Behavior**
    What you expected to happen.

    **Environment**
    - Ex-Fuzzy version:
    - Python version:
    - Operating System:

    **Additional Context**
    Any other context about the problem.

.. _suggesting-features:

Suggesting Features
===================

We welcome feature suggestions! Before suggesting:

1. **Check existing issues** and discussions
2. **Consider the scope** - does it fit Ex-Fuzzy's goals?
3. **Think about implementation** - is it feasible?

When suggesting a feature:

- Describe the use case clearly
- Explain why it would be valuable
- Provide examples if possible
- Consider backward compatibility

.. _documentation:

Documentation Contributions
============================

Documentation is crucial for user adoption. You can help by:

**Improving Existing Docs**
  - Fix typos and grammar
  - Clarify confusing sections
  - Add missing information
  - Update outdated content

**Adding New Content**
  - Write tutorials and guides
  - Create examples and case studies
  - Document best practices
  - Add API documentation

**Building Documentation**

To build the documentation locally:

.. code-block:: bash

    cd docs/
    make html
    # Open docs/_build/html/index.html

.. _code-contributions:

Code Contributions
==================

Areas where we especially welcome contributions:

**Bug Fixes**
  - Fix reported issues
  - Improve error handling
  - Address edge cases

**Performance Improvements**
  - Optimize algorithms
  - Reduce memory usage
  - Parallelize computations

**New Features**
  - Additional fuzzy set types
  - New optimization algorithms
  - Enhanced visualizations
  - Integration with other libraries

**Testing**
  - Increase test coverage
  - Add integration tests
  - Improve test performance

Code Review Process
===================

All submissions go through code review:

1. **Automated checks** run first (tests, linting, type checking)
2. **Manual review** by maintainers
3. **Discussion** and feedback
4. **Approval** and merge

What reviewers look for:

- Code correctness and efficiency
- Test coverage and quality
- Documentation completeness
- Backward compatibility
- Code style compliance

Release Process
===============

Ex-Fuzzy follows semantic versioning:

- **Major** (X.0.0): Breaking changes
- **Minor** (X.Y.0): New features, backward compatible
- **Patch** (X.Y.Z): Bug fixes, backward compatible

Release schedule:

- **Minor releases**: Every 3-4 months
- **Patch releases**: As needed for critical bugs
- **Major releases**: Annually or when significant changes accumulate

Recognition
===========

Contributors are recognized in several ways:

- **CONTRIBUTORS.md** file listing all contributors
- **Release notes** mentioning significant contributions
- **GitHub releases** crediting contributors
- **Social media** acknowledgments for major contributions

Community Guidelines
====================

We are committed to providing a welcoming environment:

**Be Respectful**
  - Treat everyone with respect
  - Welcome newcomers
  - Be patient with questions

**Be Constructive**
  - Provide helpful feedback
  - Focus on the code, not the person
  - Suggest improvements

**Be Collaborative**
  - Work together toward common goals
  - Share knowledge and resources
  - Help others learn and grow

Getting Help
============

If you need help contributing:

- **GitHub Discussions**: Ask questions and get help
- **Discord/Slack**: Real-time chat with contributors
- **Documentation**: Read our detailed guides
- **Mentorship**: We can pair new contributors with experienced ones

Thank You!
==========

Thank you for considering contributing to Ex-Fuzzy! Every contribution, no matter how small, helps make the library better for everyone.

Ready to contribute? Check out our `good first issues <https://github.com/your-username/ex-fuzzy/labels/good%20first%20issue>`_ to get started!
