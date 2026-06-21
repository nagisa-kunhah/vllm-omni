# NAVA Online Serving

This documents the expected online serving shape for `NAVAPipeline`. The
text/image-conditioned audio-video and speaker timbre paths are wired for
real-checkpoint E2E validation. Speaker references require local ReDimNet
assets prepared by the download script; runtime inference does not fetch
speaker code.

Start a NAVA server:

```bash
MODEL=/models/nava bash examples/online_serving/nava/run_server.sh
```

Submit a request:

```bash
bash examples/online_serving/nava/run_curl_nava.sh
```

The `/v1/videos` form request can pass `extra_params` as JSON for NAVA
sampling controls such as `num_frames`, `fps`, `video_guidance_scale`,
and `audio_guidance_scale`.
