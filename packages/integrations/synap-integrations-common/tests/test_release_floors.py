"""The release guard for these packages, pinned in the package it protects.

`.github/scripts/check_integration_floors.py` is what stops a PyPI release from
shipping an integration whose declared floors cannot be met. It runs once per
release, by hand, which is the worst possible place to discover that it has
rotted: the person running it is mid-release and the failure looks like the
workflow being broken rather than the guard doing its job.

So the two things that can rot are pinned here instead:

1. **The guard's package list must match the workflow's.** They are two separate
   hand-maintained lists of the same 22 folders. `synap-mcp-server` was missing
   from the workflow's list for its entire life, which is why
   `maximem-synap-mcp-server` has never been on PyPI.

2. **The guard must still catch the case it was written for.** On 2026-09-28,
   twenty integrations imported `synap_integrations_common.stream_events` while
   PyPI served a `maximem-synap-integrations-common` from before that module
   existed. Publishing them would have put twenty import-time failures on PyPI.
   That scenario is reproduced below with a fake PyPI, so the guard is tested on
   the thing that actually happened rather than on a tidier invention.

Nothing here touches the network: every test injects its own version lookup.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / ".github" / "scripts" / "check_integration_floors.py"
_WORKFLOW = _ROOT / ".github" / "workflows" / "publish-pypi-integrations.yml"


def _load_guard():
    if not _SCRIPT.exists():
        pytest.fail(f"the release guard is missing: {_SCRIPT}")
    spec = importlib.util.spec_from_file_location("check_integration_floors", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def _workflow_text() -> str:
    if not _WORKFLOW.exists():
        pytest.fail(f"the publish workflow is missing: {_WORKFLOW}")
    return _WORKFLOW.read_text()


class TestTheGuardAndTheWorkflowAgree:
    """Two hand-kept lists of the same folders. They drift silently."""

    def test_every_folder_the_workflow_publishes_is_known_to_the_guard(self):
        published = set(re.findall(r'publish_pkg "(synap-[\w-]+)"', _workflow_text()))
        assert published, "found no publish_pkg calls, so this test proves nothing"
        missing = published - set(guard.SELECTION_ENV.values())
        assert not missing, (
            f"the workflow publishes {sorted(missing)} but the guard does not know "
            "about them, so their floors are never checked"
        )

    def test_every_folder_the_guard_knows_is_published_by_the_workflow(self):
        published = set(re.findall(r'publish_pkg "(synap-[\w-]+)"', _workflow_text()))
        stale = set(guard.SELECTION_ENV.values()) - published
        assert not stale, (
            f"the guard lists {sorted(stale)} but the workflow never publishes them"
        )

    def test_every_env_var_the_guard_reads_is_passed_to_it(self):
        # The guard reads PUB_* out of the environment. A slot added to the
        # workflow's publish step but not to the guard's env block would leave
        # the guard reading "false" and checking nothing, while the package
        # publishes anyway. That is a silent hole, so pin it.
        text = _workflow_text()
        guard_step = text.split("check_integration_floors.py")[0]
        for env_name in guard.SELECTION_ENV:
            assert f"{env_name}:" in guard_step, (
                f"{env_name} is never passed to the floors check, so that package "
                "would publish unchecked"
            )

    def test_a_package_is_published_after_the_siblings_it_depends_on(self):
        """The floors check proves the dependency will exist by the END of the
        run. This pins the order WITHIN the run.

        Without it, integrations-common publishes fourth and the three packages
        ahead of it sit on PyPI for the rest of the run declaring a version of it
        that is not there yet. Anyone installing in that window gets a resolver
        error, which is a worse failure than the one the floors fixed because it
        is intermittent.
        """
        order = re.findall(r'publish_pkg "(synap-[\w-]+)"', _workflow_text())
        assert order, "found no publish_pkg calls, so this test proves nothing"
        position = {folder: index for index, folder in enumerate(order)}
        folder_of = {
            guard.distribution_name(f): f for f in guard.SELECTION_ENV.values()
        }

        wrong = []
        for folder in order:
            for requirement in guard.requirements_of(folder):
                sibling = folder_of.get(requirement.name)
                if sibling is None or sibling == folder:
                    continue  # the SDK, published by its own workflow
                if position[sibling] > position[folder]:
                    wrong.append(
                        f"{folder} (position {position[folder]}) depends on "
                        f"{sibling} (position {position[sibling]})"
                    )
        assert wrong == [], (
            "these publish before a sibling they depend on:\n  " + "\n  ".join(wrong)
        )

    def test_every_integration_with_a_pyproject_has_a_publish_slot(self):
        # The check that would have found synap-mcp-server: a package can exist,
        # build, and be tested for months with no way to release it.
        on_disk = {
            path.parent.name
            for path in (_ROOT / "integrations").glob("synap-*/pyproject.toml")
        }
        missing = on_disk - set(guard.SELECTION_ENV.values())
        assert not missing, (
            f"{sorted(missing)} have a pyproject.toml but no slot in "
            "publish-pypi-integrations.yml, so they can never be published"
        )


class TestTheCaseTheGuardWasWrittenFor:
    """2026-09-28: twenty packages about to ship against a stale sibling."""

    # Twenty integrations now import `synap_integrations_common.stream_events`,
    # so they declare a floor on the release that first carries it.
    FLOOR = "0.2.1"

    def _pypi_on_the_day(self, name: str) -> str | None:
        return {
            "maximem-synap": "0.5.3",
            "maximem-synap-integrations-common": "0.2.0",
            "maximem-synap-langchain": "0.3.1",
        }.get(name)

    def _exists_on_the_day(self, name: str, version: str) -> bool:
        """Whether that exact version was on PyPI on the day.

        Injected everywhere, or these tests reach the real PyPI. Two of them
        assert a REFUSAL, so publishing the very release this suite describes
        would silently turn them green against live data and the suite would
        stop testing anything.
        """
        latest = self._pypi_on_the_day(name)
        if latest is None:
            return False
        return guard.parse_version(version) <= guard.parse_version(latest)

    def test_publishing_an_integration_without_the_common_package_is_refused(self):
        problems = guard.check(
            ["synap-agno"],
            "patch",
            lookup=self._pypi_on_the_day,
            exists=self._exists_on_the_day,
        )
        assert problems, (
            "agno declares a floor on integrations-common that PyPI does not "
            "meet, and the guard let it through"
        )
        assert any(
            p.requirement.name == "maximem-synap-integrations-common"
            for p in problems
        )

    def test_publishing_it_together_with_the_common_package_is_allowed(self):
        # The bump takes PyPI's 0.2.0 to 0.2.1, which is exactly the floor.
        problems = guard.check(
            ["synap-agno", "synap-integrations-common"],
            "patch",
            lookup=self._pypi_on_the_day,
            exists=self._exists_on_the_day,
        )
        assert problems == [], f"expected a clean run, got {problems}"

    def test_the_whole_real_release_is_allowed(self):
        # Every folder, which is what the actual release ticks. If this fails,
        # the release itself is not safe to run.
        problems = guard.check(
            list(guard.SELECTION_ENV.values()),
            "patch",
            lookup=self._pypi_on_the_day,
            exists=self._exists_on_the_day,
        )
        assert problems == [], f"the real release would be refused: {problems}"

    def test_a_floor_on_an_unreleased_sdk_is_refused(self):
        # The SDK is published by a different workflow, so a floor on a version
        # that is not out yet can only be satisfied by publishing that first.
        def sdk_is_behind(name: str) -> str | None:
            return {
                "maximem-synap": "0.5.2",
                "maximem-synap-integrations-common": "0.2.0",
            }.get(name)

        # `exists` injected too, or this reaches the real PyPI and now passes,
        # because 0.5.3 was published while this suite was being written.
        problems = guard.check(
            ["synap-agno", "synap-integrations-common"],
            "patch",
            lookup=sdk_is_behind,
            exists=lambda name, version: False,
        )
        assert any(p.requirement.name == "maximem-synap" for p in problems), (
            "the SDK floor is 0.5.3 and PyPI has 0.5.2, which the guard must refuse"
        )

    def test_a_dependency_missing_from_pypi_entirely_is_refused(self):
        problems = guard.check(
            ["synap-agno"], "patch", lookup=lambda name: None, exists=lambda n, v: False
        )
        assert problems
        assert all(p.available is None for p in problems)


class TestPypiSummaryEndpointLag:
    """The guard runs in the minutes right after the SDK is published, which is
    exactly when PyPI's summary endpoint is stale.

    Measured on 2026-09-28: `info.version` said 0.5.2 while
    `/pypi/maximem-synap/0.5.3/json` already answered 200. Without the
    per-version fallback the guard refuses the release it exists to let through,
    which teaches the operator to distrust it.
    """

    def _stale_summary(self, name: str) -> str | None:
        return {
            "maximem-synap": "0.5.2",  # actually 0.5.3, the cache has not caught up
            "maximem-synap-integrations-common": "0.2.1",
            "maximem-synap-langchain": "0.3.1",
        }.get(name)

    def test_a_stale_summary_does_not_refuse_a_release_that_is_actually_fine(self):
        problems = guard.check(
            ["synap-agno"],
            "patch",
            lookup=self._stale_summary,
            exists=lambda name, version: (name, version) == ("maximem-synap", "0.5.3"),
        )
        assert problems == [], (
            f"the floor IS met, PyPI's summary is just behind; got {problems}"
        )

    def test_a_floor_that_was_never_published_is_still_refused(self):
        # The fallback must not turn the guard into a rubber stamp: when the
        # version genuinely does not exist, the refusal has to stand.
        problems = guard.check(
            ["synap-agno"],
            "patch",
            lookup=self._stale_summary,
            exists=lambda name, version: False,
        )
        assert any(p.requirement.name == "maximem-synap" for p in problems)


class TestTheBumpArithmeticMatchesTheWorkflow:
    """The guard predicts the version the workflow will mint. If the two
    disagree the guard is checking a version nobody publishes."""

    @pytest.mark.parametrize(
        "version,kind,expected",
        [
            ("0.2.0", "patch", "0.2.1"),
            ("0.2.0", "minor", "0.3.0"),
            ("0.2.0", "major", "1.0.0"),
            ("0.5.9", "patch", "0.5.10"),
            ("1.9.9", "minor", "1.10.0"),
        ],
    )
    def test_bump(self, version, kind, expected):
        assert guard.bump(version, kind) == expected

    def test_an_unknown_bump_type_is_an_error_not_a_guess(self):
        with pytest.raises(ValueError):
            guard.bump("0.2.0", "sideways")

    def test_a_non_numeric_version_is_an_error_not_a_mis_sort(self):
        # "0.2.0rc1" sorting below "0.2.0" by string compare is how a guard
        # passes on a release it should have stopped.
        with pytest.raises(ValueError):
            guard.parse_version("0.2.0rc1")


class TestTheWorkingTreeSatisfiesItsOwnFloors:
    """A floor is also a claim about this checkout, not only about PyPI.

    `py-distributors` in test.yml installs the SDK and integrations-common
    editable from this tree, then installs each integration editable on top. If
    an integration declares a floor higher than the version this tree carries,
    that install goes to PyPI for a release that does not exist yet and the job
    dies before running a single test. Raising the floors without raising the
    versions is a one-line change that turns the whole distributor suite red, so
    the two are pinned to each other here.
    """

    def _tree_version(self, distribution: str) -> str:
        if distribution == guard.SDK_NAME:
            version_file = _ROOT / "synap" / "sdk" / "python" / "maximem_synap" / "_version.py"
            found = re.search(r'__version__\s*=\s*"([^"]+)"', version_file.read_text())
            assert found, f"no __version__ in {version_file}"
            return found.group(1)
        folder = {
            guard.distribution_name(f): f for f in guard.SELECTION_ENV.values()
        }[distribution]
        return guard.local_version(folder)

    def test_every_declared_floor_is_met_by_this_checkout(self):
        unmet = []
        for folder in sorted(guard.SELECTION_ENV.values()):
            for requirement in guard.requirements_of(folder):
                have = self._tree_version(requirement.name)
                if guard.parse_version(have) < guard.parse_version(requirement.floor):
                    unmet.append(
                        f"{folder} needs {requirement.name}>={requirement.floor}, "
                        f"tree has {have}"
                    )
        assert unmet == [], (
            "this checkout does not satisfy its own dependency floors, so the "
            "py-distributors job cannot install it:\n  " + "\n  ".join(unmet)
        )


class TestFloorsAreReadOffTheRealPyprojects:
    def test_agno_declares_floors_on_both_shared_packages(self):
        names = {r.name for r in guard.requirements_of("synap-agno")}
        assert "maximem-synap" in names
        assert "maximem-synap-integrations-common" in names

    def test_a_framework_dependency_is_not_mistaken_for_a_maximem_one(self):
        names = {r.name for r in guard.requirements_of("synap-agno")}
        assert "agno" not in names

    def test_every_package_importing_stream_events_declares_a_common_floor(self):
        # The property that actually matters: if a package reaches for the
        # shared stream helpers, it has to say which release of them it needs.
        #
        # Matched as an import, not as a substring. synap-mcp-server names
        # `synap_integrations_common` in a docstring explaining why it does NOT
        # use it, and a substring search read that as a dependency.
        imports_common = re.compile(
            r"^\s*(?:from\s+synap_integrations_common[\w.]*\s+import|"
            r"import\s+synap_integrations_common)",
            re.M,
        )
        offenders = []
        for folder in sorted(guard.SELECTION_ENV.values()):
            source = _ROOT / "integrations" / folder
            uses_helpers = any(
                imports_common.search(path.read_text())
                for path in source.rglob("*.py")
                if "tests" not in path.parts
            )
            if not uses_helpers or folder == "synap-integrations-common":
                continue
            floors = {r.name: r.floor for r in guard.requirements_of(folder)}
            if "maximem-synap-integrations-common" not in floors:
                offenders.append(folder)
        assert offenders == [], (
            f"{offenders} import synap_integrations_common without declaring a "
            "floor on it, so pip may install a release that lacks the symbols"
        )
