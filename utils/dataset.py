# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import Dataset
import torch
import datasets
import json, os, random
import decord
from decord import VideoReader
decord.bridge.set_bridge("torch")  # make VideoReader.get_batch return torch tensors
from collections import defaultdict



class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TwoTextDataset(Dataset):
    """Dataset that returns two text prompts per sample for prompt-switch training.

    The dataset behaves similarly to :class:`TextDataset` but instead of a single
    prompt, it provides *two* prompts – typically the first prompt is used for the
    first segment of the video, and the second prompt is used after a temporal
    switch during training.

    Args:
        prompt_path (str): Path to a text file containing the *first* prompt for
            each sample. One prompt per line.
        switch_prompt_path (str): Path to a text file containing the *second*
            prompt for each sample. Must have the **same number of lines** as
            ``prompt_path`` so that prompts are paired 1-to-1.
    """
    def __init__(self, prompt_path: str, switch_prompt_path: str):
        # Load the first-segment prompts.
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        # Load the second-segment prompts.
        with open(switch_prompt_path, encoding="utf-8") as f:
            self.switch_prompt_list = [line.rstrip() for line in f]

        assert len(self.switch_prompt_list) == len(self.prompt_list), (
            "The two prompt files must contain the same number of lines so that "
            "each first-segment prompt is paired with exactly one second-segment prompt."
        )

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return {
            "prompts": self.prompt_list[idx],            # first-segment prompt
            "switch_prompts": self.switch_prompt_list[idx],  # second-segment prompt
            "idx": idx,
        }


class MultiTextDataset(Dataset):
    """Dataset for multi-segment prompts stored in a JSONL file.

    Each line is a JSON object, e.g.
        {"prompts": ["a cat", "a dog", "a bird"]}

    Args
    ----
    prompt_path : str
        Path to the JSONL file
    field       : str
        Name of the list-of-strings field, default "prompts"
    cache_dir   : str | None
        ``cache_dir`` passed to HF Datasets (optional)
    """

    def __init__(self, prompt_path: str, field: str = "prompts", cache_dir: str | None = None):
        self.ds = datasets.load_dataset(
            "json",
            data_files=prompt_path,
            split="train",
            cache_dir=cache_dir,
            streaming=False, 
        )

        assert len(self.ds) > 0, "JSONL is empty"
        assert field in self.ds.column_names, f"Missing field '{field}'"

        seg_len = len(self.ds[0][field])
        for i, ex in enumerate(self.ds):
            val = ex[field]
            assert isinstance(val, list), f"Line {i} field '{field}' is not a list"
            assert len(val) == seg_len,  f"Line {i} list length mismatch"

        self.field = field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "idx": idx,
            "prompts_list": self.ds[idx][self.field],  # List[str]
            "seed": self.ds[idx]["seed"] if self.ds[idx].get("seed", None) is not None else -1,
            "num": self.ds[idx]["num"] if self.ds[idx].get("num", None) is not None else -1
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data



class TwoTextVideoPairDataset(Dataset):
    """
    Per __getitem__:
      - returns P matched pairs from (as much as possible) different *base videos*
      - returns P mismatch pairs where tp comes from a different base video

    JSON format: list[dict], each dict at least has:
      {"file_path": "...mp4", "text": "...", ...}

    Important: idx0-8 may be segments of the same base video. We group by base video id.
    """

    def __init__(
        self,
        prompt_path: str,
        switch_prompt_path: str,
        video_json_path: str,
        *,
        n_pairs: int = 13,  # 240帧13段chunk
        delta_choices=(60, 120, 180),
        video_key: str = "file_path",
        text_key: str = "text",
        group_key: str | None = None,  # if your json has a stable "video_id", set group_key="video_id"
    ):
        # training prompts (aligned by idx)
        self.prompt_list = [l.rstrip() for l in open(prompt_path, encoding="utf-8")]
        self.switch_prompt_list = [l.rstrip() for l in open(switch_prompt_path, encoding="utf-8")]
        assert len(self.prompt_list) == len(self.switch_prompt_list), "prompt files must have same length"

        # json items
        self.items = json.load(open(video_json_path, "r", encoding="utf-8"))
        assert isinstance(self.items, list) and self.items, "video_json must be a non-empty JSON list"
        assert all(isinstance(x, dict) for x in self.items), "video_json must be list[dict]"

        self.video_key = video_key
        self.text_key = text_key
        self.group_key = group_key

        self.n_pairs = int(n_pairs)
        assert self.n_pairs >= 1
        self.delta_choices = tuple(int(d) for d in delta_choices)
        assert self.delta_choices, "delta_choices must be non-empty"

        # ---- group items by base video id ----
        self.group_to_items = defaultdict(list)
        self.video_ids = []  # keep first-appearance order for deterministic mod mapping

        for it in self.items:
            if self.video_key not in it:
                raise KeyError(f"json item missing key '{self.video_key}'")
            if self.text_key not in it:
                raise KeyError(f"json item missing key '{self.text_key}'")

            vid = self._get_video_id(it)
            if vid not in self.group_to_items:
                self.video_ids.append(vid)
            self.group_to_items[vid].append(it)

        assert self.video_ids, "no valid videos after grouping"

    def __len__(self):
        return len(self.prompt_list)

    def _get_video_id(self, it: dict) -> str:
        """
        Determine the 'base video id' for grouping segments.

        Default heuristic (works for your example):
          --v8TDp92Q0_0005550_0007350_0.mp4  ->  --v8TDp92Q0
        """
        if self.group_key is not None and self.group_key in it:
            return str(it[self.group_key])

        fname = os.path.basename(it[self.video_key])
        stem = os.path.splitext(fname)[0]
        return stem.split("_")[0]  # base id before the first underscore

    def _sample_pair_from_item(self, it: dict):
        """Sample one (t, tp) pair from a single video clip item."""
        vp = it[self.video_key]
        if not os.path.exists(vp):
            raise FileNotFoundError(vp)

        vr = VideoReader(vp)
        n = len(vr)

        if n <= 1:
            t = tp = 0
            delta = 0
        else:
            delta = max(1, min(int(random.choice(self.delta_choices)), n - 1))
            t = random.randint(0, n - 1 - delta)
            tp = t + delta

        frames = vr.get_batch([t, tp]).permute(0, 3, 1, 2).float() / 255.0  # [2,C,H,W]
        return frames, vp, it.get(self.text_key, ""), t, tp, delta

    def __getitem__(self, idx):
        # 1) choose P base videos deterministically by mod over base-video list
        num_vids = len(self.video_ids)
        base_pos = idx % num_vids
        vid_ids = [self.video_ids[(base_pos + k) % num_vids] for k in range(self.n_pairs)]

        # 2) for each chosen base video, randomly pick one segment item and sample a matched pair
        pair_list, gt_prompts, video_paths = [], [], []
        t_idx_list, tp_idx_list, delta_list = [], [], []

        for vid in vid_ids:
            it = random.choice(self.group_to_items[vid])  # pick a random segment within this base video
            frames, vp, gt_text, t, tp, delta = self._sample_pair_from_item(it)

            pair_list.append(frames)
            gt_prompts.append(gt_text)
            video_paths.append(vp)
            t_idx_list.append(t)
            tp_idx_list.append(tp)
            delta_list.append(delta)

        gt_pairs = torch.stack(pair_list, dim=0)  # [P,2,C,H,W]

        # 3) mismatch: for each k, choose tp from a DIFFERENT base video than vid_ids[k]
        #    We pick another index j among {0..P-1} such that vid_ids[j] != vid_ids[k].
        #    (If P>num_vids, duplicates can occur; we still try to find a different one.)
        tp_all = gt_pairs[:, 1]  # [P,C,H,W]
        mis_tp_list, mis_prompts, mis_video_paths = [], [], []

        for k in range(self.n_pairs):
            j = (k + 1) % self.n_pairs
            # advance j until base video differs (or we tried all P candidates)
            for _ in range(self.n_pairs):
                if vid_ids[j] != vid_ids[k]:
                    break
                j = (j + 1) % self.n_pairs

            mis_tp_list.append(tp_all[j])
            mis_prompts.append(gt_prompts[j])       # prompt of tp-source
            mis_video_paths.append(video_paths[j])  # video path of tp-source

        mis_tp = torch.stack(mis_tp_list, dim=0)  # [P,C,H,W]
        mismatch_pairs = torch.stack([gt_pairs[:, 0], mis_tp], dim=1)  # [P,2,C,H,W]

        return {
            # training prompts
            "prompts": self.prompt_list[idx],
            "switch_prompts": self.switch_prompt_list[idx],
            "idx": idx,

            # matched: P pairs, each from a (base) different video (as much as possible)
            "gt_pairs": gt_pairs.to(torch.bfloat16),            # [P,2,C,H,W]
            "gt_prompts": gt_prompts,        # list[str], len P (per pair's base video prompt)
            "video_paths": video_paths,      # list[str], len P

            # mismatch: tp comes from a different base video
            "mismatch_pairs": mismatch_pairs.to(torch.bfloat16),        # [P,2,C,H,W]
            "mismatch_prompts": mis_prompts,         # list[str], len P (tp-source prompt)
            "mismatch_video_paths": mis_video_paths, # list[str], len P (tp-source path)

            # debug/meta
            "t_idx": torch.tensor(t_idx_list, dtype=torch.long),     # [P]
            "tp_idx": torch.tensor(tp_idx_list, dtype=torch.long),   # [P]
            "delta": torch.tensor(delta_list, dtype=torch.long),     # [P]
            "base_video_ids": vid_ids,                               # list[str], len P
        }
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import cv2
    import numpy as np

    ds = TwoTextVideoPairDataset(
        prompt_path="longlive_models/prompts/vidprom_filtered_extended.txt",
        switch_prompt_path="longlive_models/prompts/vidprom_filtered_extended_switch.txt",
        video_json_path="/data/vjuicefs_ai_camera_jgroup_video/public_data/Video_Data/Sekai-Project/wan2_2_city/sekai_city_train.json",
        n_pairs=4,
        # delta_choices=(1, 2, 4, 8),  # 两帧间隔候选（单位：帧）
    )
    vis_folder = "z_vis_test"
    os.makedirs(vis_folder, exist_ok=True)

    batch_size = 1
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    for i in loader:
        for b in range(batch_size):
            for k in range(i['gt_pairs'][b].shape[0]):
                print("t_idx: {}, tp_idx:{}, delta:{}".format(i['t_idx'][b][k], i['tp_idx'][b][k], i['delta'][b][k]))
                print("[gt_prompts]:{}".format(i["gt_prompts"][b][0]))
                print("[mismatch_prompts]:{}".format(i["mismatch_prompts"][b][0]))
                gt_t_img = cv2.cvtColor(np.uint8(i['gt_pairs'][b][k][0].float().numpy().transpose(1, 2, 0)*255), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(vis_folder, f'gt_t_{k}.png'), gt_t_img)
                gt_tp_img = cv2.cvtColor(np.uint8(i['gt_pairs'][b][k][1].float().numpy().transpose(1, 2, 0)*255), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(vis_folder, f'gt_tp_{k}.png'), gt_tp_img)

                mis_gt_t_img = cv2.cvtColor(np.uint8(i['mismatch_pairs'][b][k][0].float().numpy().transpose(1, 2, 0)*255), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(vis_folder, f'mis_gt_t_{k}.png'), mis_gt_t_img)
                mis_gt_tp_img = cv2.cvtColor(np.uint8(i['mismatch_pairs'][b][k][1].float().numpy().transpose(1, 2, 0)*255), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(vis_folder, f'mis_gt_tp_{k}.png'), mis_gt_tp_img)
        print()
        # break