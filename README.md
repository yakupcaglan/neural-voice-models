# neural-voice-models

ONNX builds of open neural text-to-speech models, for on-device synthesis.

Everything here is converted from published checkpoints and redistributed under the
original licences. Nothing was trained here.

## German — `kokoro-de-thorsten-ep5`

| | |
|---|---|
| Architecture | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0) |
| Checkpoint | [Thorsten-Voice/Kokoro](https://huggingface.co/Thorsten-Voice/Kokoro), epoch 5 (Apache-2.0) |
| Voice data | [Thorsten-Voice dataset](https://www.thorsten-voice.de) — **CC0-1.0** |
| Sample rate | 24 000 Hz |
| Speakers | 1 (`thorsten`, male) |

The speaker recorded and released his own voice under CC0, so commercial use needs no
further permission.

### Why this build exists

Thorsten-Voice publishes PyTorch weights only. Converting them with sherpa-onnx's
`scripts/kokoro/v1.0/export_onnx.py` fails on current PyTorch for two reasons:

1. `Unknown number type: complex` — the vocoder's iSTFT uses complex tensors.
2. `Unsupported: ONNX export of operator Unfold` — framing uses `unfold`.

Both are avoided by replacing `TorchSTFT.transform` / `.inverse` with real-valued
equivalents (`conv1d` / `conv_transpose1d` overlap-add); unit-tested against
`torch.stft` / `torch.istft` to a max difference of 3e-6.

There is a third, quieter trap: `kokoro`'s loader silently loads only part of this
checkpoint. The state dict carries a `module.` prefix (trained with DataParallel) and
the modern `parametrizations.weight.original0/1` weight-norm names, while the package
expects the legacy `weight_g`/`weight_v`; `load_state_dict(strict=False)` swallows
both mismatches. The result is 235 of the decoder's 491 tensors loaded and the rest
left random — the model then emits white noise rather than speech. The export script
translates the keys (548 tensors loaded).

### Contents

```
kokoro-de-thorsten-ep5/
  model.onnx     325 MB   inputs: tokens [1,N] int64, style [1,256] float32, speed [1] float32
  voices.bin     522 KB   510 × 256 float32, little-endian (one speaker)
  tokens.txt     687 B    the standard Kokoro 178-token IPA vocabulary
```

Phonemes must follow espeak-ng's IPA conventions — that is what the checkpoint was
trained on. `ʏ` is absent from the vocabulary and should be substituted with `y`.

## Licences

Model weights Apache-2.0; the German voice recordings CC0-1.0. Attribution to
Thorsten Müller (Thorsten-Voice) and the Kokoro authors is kept in this README.
