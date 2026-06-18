# NAVA Offline Inference

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/nava>.

NAVA is a custom audio-video generation pipeline. It requires a prepared local model directory and the upstream NAVA Python package; it is not loaded directly from a standard Diffusers repository.

## Example README

--8<-- "examples/offline_inference/nava/README.md"

## Example materials

??? abstract "download_nava.py"
    ``````py
    --8<-- "examples/offline_inference/nava/download_nava.py"
    ``````

??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/nava/end2end.py"
    ``````
