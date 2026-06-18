# NAVA Online Serving

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/nava>.

NAVA serving uses the diffusion video generation endpoint with `NAVAPipeline`. The model path must point to a prepared local NAVA directory.

## Example README

--8<-- "examples/online_serving/nava/README.md"

## Example materials

??? abstract "run_server.sh"
    ``````bash
    --8<-- "examples/online_serving/nava/run_server.sh"
    ``````

??? abstract "run_curl_nava.sh"
    ``````bash
    --8<-- "examples/online_serving/nava/run_curl_nava.sh"
    ``````
