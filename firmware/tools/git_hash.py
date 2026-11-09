"""PlatformIO pre-script: bake the first 8 hex digits of the git commit into the firmware."""

import subprocess

Import('env')  # noqa: F821  (provided by PlatformIO/SCons)

try:
    sha = subprocess.check_output(
        ['git', 'rev-parse', '--short=8', 'HEAD'], cwd=env['PROJECT_DIR'],  # noqa: F821
        text=True).strip()
    value = int(sha, 16)
except (OSError, subprocess.CalledProcessError, ValueError):
    value = 0  # not a git checkout: PONG reports 00000000, which the host logs as unknown
env.Append(CPPDEFINES=[('ORION_FW_GIT_HASH', f'0x{value:08x}u')])  # noqa: F821
