import math
from functools import lru_cache

import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model


_SOURCE_INDEX = {
    "drums": 0,
    "bass": 1,
    "other": 2,
    "vocals": 3,
}


def _parse_time_to_seconds(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"无效时间格式: {value}") from e
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    raise ValueError(f"无效时间格式: {value}")


def _clone_audio(audio, waveform):
    return {"waveform": waveform, "sample_rate": int(audio["sample_rate"])}


@lru_cache(maxsize=2)
def _get_demucs_model(device_name: str):
    model = get_model("htdemucs")
    model.eval()
    model.to(device_name)
    return model


class AudioCropCompat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": ("STRING", {"default": "0:00"}),
                "end_time": ("STRING", {"default": "0:08"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "crop"
    CATEGORY = "audio"

    def crop(self, audio, start_time, end_time):
        sr = int(audio["sample_rate"])
        waveform = audio["waveform"]
        start_seconds = _parse_time_to_seconds(start_time) or 0.0
        end_seconds = _parse_time_to_seconds(end_time)
        start_idx = max(0, int(round(start_seconds * sr)))
        total = waveform.shape[-1]
        if end_seconds is None:
            end_idx = total
        else:
            end_idx = max(start_idx, min(total, int(round(end_seconds * sr))))
        cropped = waveform[..., start_idx:end_idx].clone()
        return (_clone_audio(audio, cropped),)


class AudioSeparationCompat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "chunk_fade_shape": (["linear"], {"default": "linear"}),
                "chunk_length": ("INT", {"default": 10, "min": 1, "max": 120, "step": 1}),
                "chunk_overlap": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.95, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("Bass", "Drums", "Other", "Vocals")
    FUNCTION = "separate"
    CATEGORY = "audio"

    def _prepare_waveform(self, waveform, sample_rate, model_sr):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        elif waveform.shape[1] > 2:
            waveform = waveform[:, :2, :]
        if sample_rate != model_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, model_sr)
        return waveform

    def _extract_source(self, separated, name, input_sr, model_sr):
        idx = _SOURCE_INDEX[name]
        stem = separated[:, idx, :, :].detach().cpu()
        if model_sr != input_sr:
            stem = torchaudio.functional.resample(stem, model_sr, input_sr)
        return stem

    def separate(self, audio, chunk_fade_shape, chunk_length, chunk_overlap):
        del chunk_fade_shape
        input_sr = int(audio["sample_rate"])
        waveform = audio["waveform"].detach().cpu().float()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _get_demucs_model(device)
        model_sr = int(model.samplerate)
        prepared = self._prepare_waveform(waveform, input_sr, model_sr)

        separated_batches = []
        with torch.inference_mode():
            for batch_idx in range(prepared.shape[0]):
                mix = prepared[batch_idx].to(device)
                out = apply_model(
                    model,
                    mix,
                    shifts=1,
                    split=True,
                    overlap=float(chunk_overlap),
                    transition_power=1.0,
                    progress=False,
                    device=device,
                    segment=float(chunk_length),
                )
                separated_batches.append(out.unsqueeze(0).cpu())
        separated = torch.cat(separated_batches, dim=0)

        bass = self._extract_source(separated, "bass", input_sr, model_sr)
        drums = self._extract_source(separated, "drums", input_sr, model_sr)
        other = self._extract_source(separated, "other", input_sr, model_sr)
        vocals = self._extract_source(separated, "vocals", input_sr, model_sr)

        return (
            _clone_audio(audio, bass),
            _clone_audio(audio, drums),
            _clone_audio(audio, other),
            _clone_audio(audio, vocals),
        )


NODE_CLASS_MAPPINGS = {
    "AudioCrop": AudioCropCompat,
    "AudioSeparation": AudioSeparationCompat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioCrop": "AudioCrop",
    "AudioSeparation": "AudioSeparation",
}
