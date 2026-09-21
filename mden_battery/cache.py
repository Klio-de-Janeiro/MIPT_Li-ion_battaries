"""Delete only recognized, reproducible NPY generations, never raw/runs.

An OS lock protects generations used by this version's dataset. Older clients
without this lock must be stopped before cleanup. Unknown files/symlinks make
a generation ineligible. Concurrent use of one generation in separate
processes is deliberately rejected (different caches are independent).
"""
import json
import os
from pathlib import Path
import re
import shutil

_OPEN = {}
_GEN = re.compile(r'generation_[0-9a-f]{12}')
_ARRAY = re.compile(r'P\d+_\d+(?:\.raw)?\.npy')
_META = {'manifest.csv', 'prepared_index.csv', 'scaler.json', 'preprocessing.json',
         '.mden_generation.json', '.mden.lock'}


class GenerationLease:
    def __init__(self, generation):
        self.path = Path(generation).resolve()
        self.closed = True
        if self.path in _OPEN:
            _OPEN[self.path][1] += 1
        else:
            stream = (self.path / '.mden.lock').open('a+b')
            try:
                if os.name == 'nt':
                    import msvcrt
                    if stream.seek(0, 2) == 0:
                        stream.write(b'0'); stream.flush()
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                stream.close()
                raise RuntimeError(f'Prepared generation is in use: {self.path}') from exc
            _OPEN[self.path] = [stream, 1]
        self.closed = False

    def close(self):
        if not self.closed:
            item = _OPEN[self.path]
            item[1] -= 1
            if item[1] == 0:
                item[0].close()  # OS releases the lock, also on process exit
                del _OPEN[self.path]
            self.closed = True

    def __del__(self):
        self.close()


def _recognized(path):
    if path.is_symlink() or not path.is_dir() or not _GEN.fullmatch(path.name):
        return False
    try:
        marker = path / '.mden_generation.json'
        if marker.exists():
            if json.loads(marker.read_text()).get('owner') != 'mden_prepared_cache':
                return False
        else:
            # Audited v8 generations created before the ownership marker.
            meta = json.loads((path / 'preprocessing.json').read_text())
            if meta.get('version') != 8 or meta.get('columns') != [
                    'cycle', 'time_s', 'voltage_V', 'current_A', 'temperature_C', 'soc', 'soh']:
                return False
            if not all((path / f).is_file() for f in ('manifest.csv', 'prepared_index.csv', 'scaler.json')):
                return False
        for child in path.iterdir():
            if child.is_symlink():
                return False
            if child.name == 'arrays' and child.is_dir():
                if any(p.is_symlink() or not p.is_file() or not _ARRAY.fullmatch(p.name) for p in child.iterdir()):
                    return False
            elif not child.is_file() or child.name.removesuffix('.tmp') not in _META:
                return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def cleanup_prepared_cache(root, *, include_current=False, dry_run=True):
    """Prune direct generation children only. Default previews deletion; a small lock file may be created."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Refusing a symlink cache root')
    root = root.resolve()
    pointer = root / 'CURRENT.json'
    if pointer.is_symlink():
        raise ValueError('Refusing a symlink CURRENT pointer')
    active = json.loads(pointer.read_text())['generation'] if pointer.is_file() else None
    if active is not None and not _GEN.fullmatch(active):
        raise ValueError('Invalid CURRENT generation')
    records = []
    if not root.is_dir():
        return records
    for path in sorted(root.iterdir()):
        if not _GEN.fullmatch(path.name):
            continue
        if path.name == active and not include_current:
            continue
        if path.resolve() in _OPEN or not _recognized(path):
            records.append(dict(generation=path.name, status='skipped_in_use_or_unrecognized'))
            continue
        lease = None
        try:
            lease = GenerationLease(path)
            # Recheck after acquiring lock, including symlinks and added files.
            if not _recognized(path):
                records.append(dict(generation=path.name, status='skipped_unrecognized'))
                continue
            size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
            if not dry_run:
                # Keep lock file until large arrays are gone (Windows permits
                # removing other files while the lock handle remains open).
                for child in path.iterdir():
                    if child.name == '.mden.lock':
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                lease.close()
                (path / '.mden.lock').unlink()
                path.rmdir()
                if path.name == active and pointer.is_file():
                    # Do not remove a pointer published to another generation.
                    if json.loads(pointer.read_text()).get('generation') == path.name:
                        pointer.unlink()
            records.append(dict(generation=path.name, status='would_delete' if dry_run else 'deleted', bytes=size))
        except RuntimeError:
            records.append(dict(generation=path.name, status='skipped_in_use'))
        finally:
            if lease is not None:
                lease.close()
    return records


def save_data_recipe(generation, run_dir):
    """Keep small provenance files so an NPY cache can be safely regenerated."""
    destination = Path(run_dir) / 'data_recipe'
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.csv', 'prepared_index.csv', 'scaler.json', 'preprocessing.json'):
        shutil.copy2(Path(generation) / name, destination / name)
    return destination
