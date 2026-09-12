# DeepSeek-V4-Flash-Vision-Exp model-author runtime

The pinned vLLM oracle implements the DeepSeek-V4 text architecture but not the
vision variant. Code searches on 2026-08-31 found no complete implementation of
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at vLLM main
`dafbef15a1c879c64ebb99427917e4ca8d5bca1e`, vLLM-Omni main
`b81aeb7b86837f6fe8956f3aef83798ad26c5a26` or SGLang main
`52e1c24744bf4efe75fe976e26596ae1c9f279e2`. Transformers main
`a3f3da8f87dc65d724d500eeb44777e4716aaa46` contains the DeepSeek-V4 text
model but not this checkpoint repository's vision fields or vision forward.

The model-author repository is therefore the secondary oracle only for behavior
vLLM does not define: its prompt image blocks, image processor, 32-layer ViT,
aligner, learned image sentinels and image-span attention visibility. vLLM stays
the primary oracle for the language backbone, KV cache, MoE, MHC, sampling and
serving behavior it implements.

The repository is a Hugging Face Git repository. Revision
`86f746b36186f0e567729a5c06a8c918caba82a9` was read on 2026-08-31. It carries
`encoding/encoding_dsv4.py`, `inference/image_processor.py`,
`inference/vision.py`, `inference/model.py`, `inference/convert.py` and the
48-shard checkpoint index. The index reports `167,811,372,792` bytes. The model
card's minimal-inference recipe converts and runs the model at tensor parallel
size 4.

`gateable = no`. This session read the source and checkpoint index but did not
install the runtime, load the 156.287 GiB artifact or generate a token. A source
read and a constructed config are not a run. Issue #2411 owns the first leased
TP4 build, full-index validation, two-image generation and committed stage/output
evidence.

The model card tells users to run `inference/test_image_processor.py`, but that
file returns HTTP 404 at this pin and is absent from the repository tree. This
missing test does not weaken the port. The first oracle run executes the pinned
`image_processor.py` on committed fixtures and records its outputs directly.

```oracle-pin
id = deepseek-v4-vision
role = secondary
upstream = https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
scope = prompt encoding, image preprocessing, ViT, aligner, learned image sentinels and image-span visibility for DeepSeek-V4-Flash-Vision-Exp, which vLLM does not implement
pin = 86f746b36186f0e567729a5c06a8c918caba82a9
pin_label = Hugging Face revision 86f746b3
pinned_on = 2026-08-31
gateable = no
evidence = #2411
```
