# Third-party weapon model

Source: https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8

Pinned source commit: `3d641e6001abdaa1afa3cd45d0fe02a9554f3d0e`

Artifact: `models/normal.onnx`, installed locally as `models/weapons.onnx`.

SHA-256: `910d724547692ff4ba75ff4f335b0e72344c2b5382bbbd5ab08e9ba4cc3dfc0b`

Embedded class mapping: index 0 `guns`, index 1 `knife`.

Input: `[1, 3, 640, 640]`. Output: `[1, 6, 8400]`. Metadata identifies Ultralytics 8.0.230 and a detection model trained on `model/data.yaml`.

## Conflicting license declarations

The upstream README claims MIT, but its actual LICENSE file contains GPL-3.0; an unchanged copy is retained as `LICENSE.upstream`. The ONNX metadata declares `AGPL-3.0 https://ultralytics.com/license`.

Do not treat these weights as MIT-licensed. AGPL-3.0 is the model's embedded license declaration, with GPL-3.0 also present upstream. Clarify applicable terms with the model publisher and Ultralytics before operational/commercial deployment or redistribution. No MIT or proprietary-use permission is asserted here.

The weights are excluded from this project's Git repository. The download helper retrieves them directly from the pinned upstream source for local evaluation, retaining attribution and this notice. Upstream training code, checkpoint weights and training records remain available at the source link. Dataset provenance is the Roboflow dataset linked by that upstream README; its rights have not been independently verified.

## Validation scope

ONNX structural validation, embedded label inspection and actual OpenCV inference passed. Repository demonstration/validation images can exercise detections but are not an independent CCTV benchmark. Model performance has not been validated on the user's camera footage; false alarms and missed events remain possible. Detecting an object does not identify who carries it or establish violent intent.
