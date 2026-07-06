import os
import sys


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)


# FoocusRX change: auto-update is now OPT-IN.
#
# Upstream Fooocus force-updated the working copy on every launch via
# `git fetch` + hard reset. That silently wiped local edits to config.txt,
# preset JSON, or any .py file the user was iterating on. It also disabled
# git's dubious-ownership check globally.
#
# In FoocusRX auto-update runs only when the user passes --auto-update
# (or sets FOOOCUS_AUTO_UPDATE=1), and even then refuses to touch a dirty
# working tree.
_AUTO_UPDATE = ("--auto-update" in sys.argv) or (os.environ.get("FOOOCUS_AUTO_UPDATE") == "1")
# Strip the flag so it doesn't confuse the downstream argparse in launch.py.
if "--auto-update" in sys.argv:
    sys.argv.remove("--auto-update")


def _try_auto_update():
    try:
        import pygit2
    except Exception as e:  # pygit2 missing on a fresh env — skip cleanly.
        print(f"Auto-update skipped: pygit2 unavailable ({e}).")
        return

    try:
        repo = pygit2.Repository(root)

        # Refuse to overwrite user modifications.
        status = repo.status()
        # Ignore ignored/untracked-only files; only bail on modifications
        # to tracked files that a hard-reset would clobber.
        DIRTY = (
            pygit2.GIT_STATUS_WT_MODIFIED
            | pygit2.GIT_STATUS_WT_DELETED
            | pygit2.GIT_STATUS_WT_RENAMED
            | pygit2.GIT_STATUS_WT_TYPECHANGE
            | pygit2.GIT_STATUS_INDEX_MODIFIED
            | pygit2.GIT_STATUS_INDEX_NEW
            | pygit2.GIT_STATUS_INDEX_DELETED
            | pygit2.GIT_STATUS_INDEX_RENAMED
            | pygit2.GIT_STATUS_INDEX_TYPECHANGE
            | pygit2.GIT_STATUS_CONFLICTED
        )
        dirty_files = [p for p, flags in status.items() if flags & DIRTY]
        if dirty_files:
            preview = ", ".join(dirty_files[:5]) + ("..." if len(dirty_files) > 5 else "")
            print(
                f"Auto-update skipped: working tree has local modifications "
                f"({len(dirty_files)} file(s): {preview}). "
                f"Commit, stash, or discard them and re-run."
            )
            return

        branch_name = repo.head.shorthand
        remote_name = "origin"
        remote = repo.remotes[remote_name]

        remote.fetch()

        local_branch_ref = f"refs/heads/{branch_name}"
        local_branch = repo.lookup_reference(local_branch_ref)

        remote_reference = f"refs/remotes/{remote_name}/{branch_name}"
        remote_commit = repo.revparse_single(remote_reference)

        merge_result, _ = repo.merge_analysis(remote_commit.id)

        if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
            print("Already up-to-date.")
        elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
            local_branch.set_target(remote_commit.id)
            repo.head.set_target(remote_commit.id)
            repo.checkout_tree(repo.get(remote_commit.id))
            repo.reset(local_branch.target, pygit2.GIT_RESET_HARD)
            print("Auto-update: fast-forwarded.")
        elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
            print(
                "Auto-update skipped: local branch has diverged from origin. "
                "Resolve manually with `git pull --rebase`."
            )
        else:
            print(f"Auto-update skipped: unexpected merge_analysis result {merge_result}.")
    except Exception as e:
        print("Auto-update failed:")
        print(f"  {e!r}")


if _AUTO_UPDATE:
    _try_auto_update()
else:
    print("Auto-update disabled (pass --auto-update or set FOOOCUS_AUTO_UPDATE=1 to enable).")


from launch import *
