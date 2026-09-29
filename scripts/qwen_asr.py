"""Transcribe prepared chunks locally, loading Qwen once for the recording."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import dataclasses
from importlib.metadata import version
import json
from pathlib import Path
import resource
import time

import mlx.core as mx
from mlx_audio.stt.utils import load_model

ROOT = Path(__file__).resolve().parents[1]
MAX_TOKENS = 2048


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filelist', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    files = args.filelist.read_text(encoding='utf-8').splitlines()
    if not files or any(not Path(name).is_file() for name in files):
        parser.error('Expected a nonempty list of prepared audio files')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = load_model(ROOT/'models/qwen17', strict=True)
    settings = {
        'engine': 'mlx-audio ' + version('mlx-audio'),
        'mlx_version': version('mlx'), 'precision': 'BF16',
        'language': 'Japanese', 'temperature': 0,
        'max_tokens_per_input': MAX_TOKENS, 'internal_chunk_seconds': 1200,
        'hotwords': None, 'system_prompt': None,
        'load_seconds': time.perf_counter()-started,
        'time_granularity': 'input chunks only; not forced aligned',
        'audio_upload': False,
    }
    save(args.output/'settings.json', settings)
    for index, audio in enumerate(files):
        started = time.perf_counter()
        result = model.generate(audio, language='Japanese', temperature=0,
                                max_tokens=MAX_TOKENS, chunk_duration=1200,
                                verbose=False)
        raw = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else vars(result)
        record = {
            'file': audio, 'raw': raw, 'processing_seconds': time.perf_counter()-started,
            'token_limit_reached': result.generation_tokens >= MAX_TOKENS,
            'process_max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'mlx_peak_memory_bytes': mx.get_peak_memory(),
            'memory_scope': 'macOS process high-water RSS and MLX allocator peak; cumulative since worker start',
        }
        save(args.output/f'{index:03d}.json', record)
        print(f'qwen_chunk {index+1}/{len(files)} {record["processing_seconds"]:.2f}s', flush=True)


if __name__ == '__main__':
    main()
