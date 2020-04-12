.. _release_procedure:

Release Procedure
=================

How to issue a xarray-simlab release in a few steps:

1. Ensure local master branch is synced to upstream::

     $ git pull upstream master

2. Check ``whats_new.rst`` and the docs. Make sure "Release Notes" is
   complete (check the date!) and if needed add a brief summary note
   describing the release at the top.

3. If you have any doubts, run the full test suite one final time!::

     $ pytest xsimlab -vv

5. On the master branch, commit the release in git::

     $ git commit -a -m 'release v0.X.Y'

6. Tag the release::

     $ git tag -a 0.X.Y -m 'release v0.X.Y'

7. Push to GitHub::

     $ git push upstream master --tags

8. Publish the release on GitHub: go to the repository's URL, follow
   the ``releases`` link, click on the ``Draft a new release`` button,
   select the tag of this release, add a title (e.g., the tag name)
   and a description (e.g., the summary added in ``whats_new.rst``).

8. Before build the package and upload to PyPI, make sure that you
   didn't make a local install using pip (maybe due to .egg-info
   conflict, this make cause issue with the packaged version on PyPI,
   which may be unusable and this is irreversible!). For steps 8 and 9
   below, either you can switch to another local clone of the
   repository (clean and up-to-date!! repeat step 1 if needed), or
   first clean the repository from build/dist files and/or all git
   untracked and ignored files (if you don't mind losing them)::

     $ rm -rf dist build */*.egg-info *.egg-info
     $ git clean -xfd

9. Build source and binary wheels for PyPI::

     $ python setup.py bdist_wheel sdist

10. Use twine to register and upload the release on pypi. You will
    need to be listed as a package owner at
    https://pypi.python.org/pypi/xarray-simlab for this to work. Be
    careful, this is irreversible!!::

      $ twine upload dist/xarray-simlab-0.X.Y*

11. Update conda-forge. Clone
    https://github.com/conda-forge/xarray-simlab-feedstock and update
    the version number and sha256 in ``recipe/meta.yaml`` (check also
    dependencies). Submit a pull request (and merge it, once CI
    passes). Note: on macOS, you can calculate sha256 with::

      $ shasum -a 256 xarray-simlab-0.X.Y.tar.gz

12. Add a Section for the next release (v.X.(Y+1)) to
    ``doc/whats-new.rst``.

13. Commit your changes and push to master again::

      $ git commit -a -m 'Revert to dev version'
      $ git push upstream master
