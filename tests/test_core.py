from vdaworld.core.api import WorldAPI
from vdaworld.utils.cache import DiskCache


def test_world_api_creates_call_directory_in_requested_output(tmp_path):
    output_dir = tmp_path / "run"
    api = WorldAPI(output_dir=str(output_dir))

    assert api.output_dir == str(output_dir)
    assert (output_dir / "api_calls").is_dir()


def test_disk_cache_does_not_create_directory_until_first_access(tmp_path):
    cache_dir = tmp_path / "run" / "_cache"
    cache = DiskCache(str(cache_dir))

    assert not cache_dir.exists()

    assert cache.get_pickle("segment", "missing") is None
    assert cache_dir.is_dir()
