"""Password-gated admin UI for editing wildlife config from the gallery.

This subpackage is intentionally isolated from the hardware-free gallery core:
its heavy/optional imports (``ruamel.yaml`` for round-trip config writes, and the
camera *probe* which pulls in ``cv2``/``reolink-aio``) are confined here and, in
the probe's case, run in a separate subprocess. See:

* :mod:`wildlife.admin.config_io` -- validated, atomic, comment-preserving writes.
* :mod:`wildlife.admin.probe`     -- isolated "test this camera" subprocess.
* :mod:`wildlife.admin.auth`      -- HTTP Basic Auth guard for the admin routes.
* :mod:`wildlife.admin.routes`    -- the Flask blueprint mounted at ``/admin``.
* :mod:`wildlife.admin.passwd`    -- ``wildlife-admin-password`` CLI.
"""
