## PyCCD Development Process
Roles

* Developer
* Technical Reviewers
* Science Reviewers
* Change Review Board - in addition to the decisions makers, we recommend the board include one or more of the following advisors:
* An advocate for developers
* A library maintainer / release manager representative
* An operations representative
* Software Library Maintainers
* Operations

Workflow

1. A Developer gets a new idea for a feature
2. The Developer forks the software repository
3. The Developer creates a feature branch from the branch named develop
  1. For each new feature, a separate branch is created from the develop branch of the software repository
  2. In other words, each branch has only one feature; if more features evolve from the branch, those need to be put into separate branches
  3. A feature is defined as an indivisible unit of functionality for the given software library
4. Upon completion of the feature, the Developer submits the branch to Technical Reviewers using the Github "Pull Request" capability
5. The Developer addresses all technical feedback from peers, but does not merge
6. The Developer submits feature descriptions and any science changes (e.g., an algorithm update) to Science Reviewers
7. The Developer addresses all science feedback from peers, but does not merge
8. The Developer submits the feature to the Change Review Board
9. The Developer addresses feedback from the Change Review Board
10. Upon completion of feedback, the  Change Review Board will either approve or deny the request to include the feature in a coming release
11. If the Change Review Board approves a feature for inclusion in a release:
  1. The Change Review Board will determine in which future release the new feature will go -- it may not be the next release
  2. If the feature will go in the next release, the developer will merge only the approved branch to develop or ask the Software Library Maintainer to perform the merge
  3. If the feature will go in a later release, the branch will not be merged until the time of the future release's "release candidate" mode
  4. Features that are not approved are not merged into any branch
12. The Change Review Board provides a list of features that will constitute a next release
13. When all the approved features are ready for release, the Software Library Maintainer will create a new release branch named with the version number for that release, at which point it becomes a release candidate
  1. Only features which have been approved by the Change Review Board are allowed in the release branch
  2. During a release, the develop branch will go into "feature freeze" mode: no changes or new features are merged into develop; however, developers may continue their work on their own branches
  3. The only exception to this is when testing shows that a release candidate feature has a bug: a fix is then applied to both the release branch and the develop branch
14. When a release candidate has been reviewed and tested, the Software Library Maintainer will merge the release branch to the master branch
15. When the release has been merged to the master branch, the Software Library Maintainer will tag master with the version number of the release
16. At this point, the feature freeze is over and the develop branch may again have new features (for the next release) merged into it
17, On the date specified by the Change Review Board, the latest release of the software will be deployed by Operations and thus made available to users of the software

#### app.py

#### cli.py and entry_point scripts
The command line interface is implemented using the click project, which
provides decorators for functions that become command line arguments.

Integration with setup.py entry_point is done via click-plugins, which allow
cli commands to also be designated as entry point scripts.

See ccd.cli.py, setup.py and the click/click-plugin documentation.

* [Click Docs](http://click.pocoo.org/5/)
* [Click On Github](https://github.com/pallets/click)
* [Click on PyPi](https://pypi.python.org/pypi/click)
* [Click-Plugins on Github](https://github.com/click-contrib/click-plugins)
* [Click-Plugins on PyPi](https://pypi.python.org/pypi/click-plugins)


#### logging
Basic Python logging is used in pyccd and is fully configured in app.py. To use logging in any module:

```python
from ccd import app

logger = app.logging.getLogger(__name__)

logger.info("Info level messages")
logger.debug("Debug code")
```

## Performance TODO
* optimize data structures (numpy)
* use pypy
* employ @lrucache
