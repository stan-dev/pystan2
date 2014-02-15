This documentation is intended for use by pystan developers.

Merging in upstream stan changes
================================

Git subtree is used to update the Stan source code. Typically this is done
infrequently.

Updating the source based on Stan git tags is simple. For instance, this is the
command needed to update the source to ``v2.2.0``:

::

    git subtree pull --prefix pystan/stan https://github.com/stan-dev/stan.git v2.2.0 --squash
