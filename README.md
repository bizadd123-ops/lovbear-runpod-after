# LOVBEAR RunPod Serverless Worker (생성형 After)

비의료·참고용 **실사 After 이미지** 생성 워커입니다.
3D 아바타(TripoSR, 별도 리포)와는 **완전히 별개**입니다.

## 배포 방식 — Docker 빌드 없음

이 리포는 RunPod **Serverless Endpoint의 GitHub 연동(Docker 빌드)** 을 쓰지
않습니다. 대신 RunPod Template(`lovbear-after`, id `tp7iti5b18`)의
`dockerStartCmd`가 컨테이너 부팅 시:

1. `pip install runpod diffusers transformers accelerate safetensors Pillow`
2. `curl`로 이 리포의 `main` 브랜치 `handler.py`를 raw로 받아옴
   (`https://raw.githubusercontent.com/bizadd123-ops/lovbear-runpod-after/main/handler.py`)
3. `python -u handler.py` 실행

즉 **이 리포의 `main`에 push만 하면** (Docker 이미지 재빌드 없이) 다음에
뜨는 워커부터 새 코드가 적용됩니다. 이미 켜져 있는 워커는 idle timeout이
지나 내려가야 새 코드를 받습니다 — 급하면 RunPod 콘솔에서 워커를 수동으로
내렸다가 다시 올리면 됩니다.

Endpoint: `lovbear-generative-after` (id `scndt158vq7b2h`) — 앱의
`RUNPOD_ENDPOINT_ID` 환경변수가 이미 이 값으로 설정되어 있습니다.

## 요청 형식

앱이 `POST /api/generate-after` → RunPod `/run` 으로 보냅니다.

```json
{
  "input": {
    "task": "inpaint_edit",
    "image_base64": "...",
    "mask_base64": "...",
    "prompt": "...",
    "negative_prompt": "...",
    "strength": 0.35,
    "procedure_id": "alar",
    "purpose": "non_medical_image_simulation"
  }
}
```

- `mask_base64`가 있으면 **국소 인페인팅**(마스크 안쪽만 재생성, 바깥쪽은
  원본 픽셀 그대로 합성)을 씁니다. 웹 앱은 얼굴 랜드마크 기반으로 이 마스크를
  만들어 항상 같이 보냅니다 (`src/lib/face/ai-mask.ts`) — 그래서 "코"를
  프롬프트에 넣어도 코 주변만 바뀌고 나머지 얼굴은 왜곡되지 않습니다.
- `mask_base64`가 없으면 예전처럼 전체 이미지 img2img로 폴백합니다
  (하위 호환용, 지금 웹 앱은 항상 마스크를 보냄).

## 응답

```json
{ "image_base64": "...", "mode": "inpaint" }
```

## 환경변수 (RunPod Template)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_ID` | `stabilityai/stable-diffusion-xl-base-1.0` | 마스크 없을 때 img2img용 |
| `INPAINT_MODEL_ID` | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | 마스크 있을 때 인페인팅용 (전용 체크포인트라 품질이 더 좋음) |
| `MAX_SIDE` | `768` | 작업 해상도 (긴 변 기준) |
| `STEPS` | `20` | 추론 스텝 |
| `GUIDANCE` | `6.5` | CFG scale |

## 품질 업 (선택)

- IP-Adapter / InstantID / FaceID로 정체성 보존 강화
- ControlNet(openpose/softedge)로 구조 유지
- 자체 LoRA

## 비용

Serverless는 실행 시간(초) × GPU 단가입니다. 요청 없으면 거의 0에 가깝습니다
(콜드스타트 시 첫 요청이 느릴 수 있음).

## 고지

의료·시술 권고가 아닙니다. 이미지는 시뮬레이션 목적만 사용하세요.
