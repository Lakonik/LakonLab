import argparse
import os
import orjson
import gzip
import webdataset as wds
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import get_token
from mmcv.fileio.file_client import HardDiskBackend
from torch.utils.data import DataLoader
from lakonlab.utils.io_utils import FileClient
from lakonlab.utils import download_from_huggingface

# tar shards
N_SHARDS = 300
SHARD_PATTERN = "data/train-{i:05d}-of-00300.tar"
BASE_URL = "https://huggingface.co/datasets/Lakonik/laion-3m/resolve/main/"
PROMPT_DATASET = "Lakonik/t2i-prompts-3m"
PROMPT_MAPPING_URL = "huggingface://datasets/Lakonik/t2i-prompts-3m/prompt_mapping.jsonl.gz"


def parse_args():
    parser = argparse.ArgumentParser(description="Download the LAION-3M dataset")
    parser.add_argument("--out-dir", default="data/laion-3m/", help="Output directory, can be a local path or an S3 URL")
    parser.add_argument("--workers", type=int, default=24, help="Number of image writer workers")
    parser.add_argument("--batch-shards", type=int, default=8, help="Number of shards to download in parallel")
    parser.add_argument("--num-shards", type=int, default=N_SHARDS, help="Number of shards to download")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be greater than 0")
    if args.batch_shards <= 0:
        parser.error("--batch-shards must be greater than 0")
    if not 1 <= args.num_shards <= N_SHARDS:
        parser.error(f"--num-shards must be between 1 and {N_SHARDS}")
    return args


def process_item(raw_bytes: bytes, filename: str, out_image_dir: str):
    fc = FileClient.infer_client(uri=out_image_dir)
    out_path = fc.join_path(out_image_dir, filename)
    fc.put(raw_bytes, out_path)


def write_text(content, path):
    fc = FileClient.infer_client(uri=path)
    if isinstance(fc.client, HardDiskBackend):
        tmp_path = path + ".tmp"
        fc.put_text(content, tmp_path)
        os.replace(tmp_path, path)  # atomic replace to avoid partial writes
    else:
        fc.put_text(content, path)


def load_prompt_mapping():
    hashlist = download_from_huggingface(PROMPT_MAPPING_URL)
    prompt_ds = load_dataset(PROMPT_DATASET)["train"]
    all_prompts = prompt_ds.to_pandas()["prompt"].tolist()

    with gzip.open(hashlist, "rt", encoding="utf-8") as f:
        m = {}
        for line, prompt in zip(f, all_prompts):
            item = orjson.loads(line)
            m[item["image_hash"]] = prompt
    return m


def load_existing_state(out_datalist_path, out_prompt_path):
    existing = set()
    out_datalist = []
    out_prompts = []

    fc_d = FileClient.infer_client(uri=out_datalist_path)
    if fc_d.exists(out_datalist_path):
        for line in fc_d.get_text(out_datalist_path).splitlines():
            item = orjson.loads(line)
            out_datalist.append(item)
            existing.add(item["filename"])

    fc_p = FileClient.infer_client(uri=out_prompt_path)
    if fc_p.exists(out_prompt_path):
        for line in fc_p.get_text(out_prompt_path).splitlines():
            out_prompts.append(orjson.loads(line))

    # keep strict consistency
    if len(out_prompts) != len(out_datalist):
        n = min(len(out_prompts), len(out_datalist))
        out_prompts = out_prompts[:n]
        out_datalist = out_datalist[:n]
        existing = set(x["filename"] for x in out_datalist)

    return existing, out_datalist, out_prompts


def flush_state(out_datalist, out_prompts, out_datalist_path, out_prompt_path):
    datalist_content = "\n".join(orjson.dumps(x).decode("utf-8") for x in out_datalist)
    write_text(datalist_content, out_datalist_path)

    prompt_content = "\n".join(orjson.dumps(x).decode("utf-8") for x in out_prompts)
    write_text(prompt_content, out_prompt_path)


def load_resume_state(resume_path):
    fc = FileClient.infer_client(uri=resume_path)
    if fc.exists(resume_path):
        try:
            return orjson.loads(fc.get_text(resume_path))
        except orjson.JSONDecodeError:
            print(f"Ignoring invalid resume state: {resume_path}")
            return {}
    return {}


def save_resume_state(state: dict, resume_path):
    write_text(
        orjson.dumps(state).decode("utf-8"),
        resume_path,
    )


def shard_pipe_url(shard_idx: int, token: str | None):
    url = BASE_URL + SHARD_PATTERN.format(i=shard_idx)
    if token:
        return f"pipe:curl -s -L '{url}' -H 'Authorization:Bearer {token}'"
    else:
        return f"pipe:curl -s -L '{url}'"


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_image_dir = os.path.join(out_dir, "images")
    out_datalist_path = os.path.join(out_dir, "images.jsonl")
    out_prompt_path = os.path.join(out_dir, "prompts.jsonl")
    resume_path = os.path.join(out_dir, "resume_state.json")
    cap = args.workers * 2

    hash_prompt_mapping = load_prompt_mapping()
    existing, out_datalist, out_prompts = load_existing_state(out_datalist_path, out_prompt_path)

    # resume by shard + offset
    rs = load_resume_state(resume_path)
    start_shard = int(rs.get("shard_idx", 0))

    hf_token = get_token()

    pool = ProcessPoolExecutor(max_workers=args.workers)
    pending = set()

    written_total = len(out_datalist)  # total committed (including previous runs)

    for shard0 in tqdm(range(start_shard, args.num_shards, args.batch_shards), desc="shard batches", position=0):
        shard1 = min(args.num_shards, shard0 + args.batch_shards)
        num_workers = shard1 - shard0

        urls = [shard_pipe_url(i, hf_token) for i in range(shard0, shard1)]
        ds = wds.WebDataset(urls, shardshuffle=False)  # multiple shards
        dl = DataLoader(
            ds,
            batch_size=None,
            num_workers=num_workers,  # parallel shard downloads
            prefetch_factor=2,
        )

        for sample in tqdm(dl, total=10000 * (shard1 - shard0),
                           desc=f"samples shards {shard0:05d}-{shard1 - 1:05d}",
                           position=1, leave=False):
            # parse meta
            meta = sample["json"]
            if isinstance(meta, (bytes, bytearray, memoryview)):
                meta = orjson.loads(bytes(meta))
            # png bytes
            raw = sample["png"]
            if isinstance(raw, memoryview):
                raw = raw.tobytes()

            image_hash = meta["image_hash"]

            if image_hash in existing:
                continue
            existing.add(image_hash)

            height, width = meta["hw"]
            bucket_id = meta["size_idx"]
            prompt = hash_prompt_mapping[image_hash]

            if len(pending) >= cap:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()
            pending.add(pool.submit(process_item, raw, image_hash + ".png", out_image_dir))

            out_datalist.append(dict(filename=image_hash, bucket_id=bucket_id, image_size=[height, width]))
            out_prompts.append(dict(prompt=prompt, height=height, width=width))

            written_total += 1

        if pending:
            done, pending = wait(pending)
            for f in done:
                f.result()

        flush_state(out_datalist, out_prompts, out_datalist_path, out_prompt_path)
        save_resume_state(dict(shard_idx=shard1, written_total=written_total), resume_path)

    pool.shutdown(wait=True)
    print(f"Exported {len(out_datalist)} items. images/prompts in {out_dir}")


if __name__ == "__main__":
    main()
