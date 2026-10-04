# Temporal fighting model

Source: https://github.com/MohamedSebaie/Fight_Detection_From_Surveillance_Cameras-PyTorch_Project

Pinned commit: `a27adffec6651c10639b5b453e0df8564b9be298`

Artifact: `Models/model_16_m3_0.8888.pth`. Installed locally as `models/violence.pth`.

SHA-256: `8268694ade4c0e79a1c5fa41a4f3c21129082d75fac9eb7c25e5a0eabad49bfa`

The source repository's MIT license is retained in `LICENSE.violence`, copyright 2021 team el ranaan. Weights are downloaded from the original source rather than redistributed here. Training/inference source and dataset records remain at the linked repository. Dataset rights have not been independently verified; repository-reported accuracy is not a CCTV deployment guarantee.

Architecture: torchvision `mc3_18`, with a two-output classifier, tensor state dict loaded with `weights_only=True`, on CPU. Input: RGB `[1,3,16,112,112]`. The upstream class order is `['fight', 'noFight']`, so output index 0 is the fighting score.

Preprocessing matches the source's resize to 128×171, center crop to 112×112, mean `[0.43216,0.394666,0.37645]`, and standard deviation `[0.22803,0.22145,0.216989]`. The configured zone is cropped first. Center cropping means activity at the edge can be missed.

This application samples up to eight frames per video second, classifies 16-frame windows, and advances by eight sampled frames. Two consecutive high-score windows plus the configured duration are required before alerting; cooldown limits repeats. Very low-frame-rate footage spans longer windows. These operational choices require validation with the intended camera footage.

Scores apply to the video window. They do not identify participants, infer anger/emotion, establish intent, or distinguish every kind of physical violence. Sports, staged fights and rough play may be confused with actual fighting. Human review is required. The evidence AVI contains the 16 sampled frames used by the model (no audio, no future frames); timestamps record the source-video interval. Playback approximates eight frames/sec and can differ from elapsed source time at low FPS.
