from copy import deepcopy
import pytest
from .verify import check_certificate
from experiments.concurrency_validation.frontier import physical_batches, certificate

def fixture():
    batches = physical_batches(16, 8)
    dependencies = {'batches': [dict(group=g, batch=b, bundles=v) for (g,b),v in sorted(batches.items())]}
    return certificate(16, 8, 1600, batches), dependencies

def test_independent_checker_accepts_and_rejects_missing_bundle():
    proof, dependencies = fixture()
    assert check_certificate(proof, dependencies) == 16
    corrupted = deepcopy(proof)
    corrupted['batches'][0]['sublayers'][0].pop()
    with pytest.raises(AssertionError):
        check_certificate(corrupted, dependencies)

def test_independent_checker_rejects_endpoint_conflicts():
    proof, dependencies = fixture()
    bundle = dependencies['batches'][0]['bundles'][1]
    original = bundle['pairs'][0]
    bundle['pairs'][0] = (dependencies['batches'][0]['bundles'][0]['pairs'][0][0], original[1])
    with pytest.raises(AssertionError):
        check_certificate(proof, dependencies)
