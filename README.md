# LOVBEAR RunPod Serverless Worker (생성형 After)

비의료·참고용 **실사 After 이미지** 생성 워커입니다.  
신경망 3D(DECA)와는 **별도**입니다.

## 1. RunPod에서 할 일

1. [RunPod](https://www.runpod.io/) 가입 → API Key 발급  
2. **Serverless → New Endpoint**  
3. GPU 선택 (예: RTX 4090 / A6000)  
4. 이 `handler.py`를 워커로 배포  
   - GitHub 연동 또는 Docker 빌드 (`Dockerfile` 참고)  
5. Endpoint ID 복사

## 2. 앱 `.env.local`

```bash
RUNPOD_API_KEY=your_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
```

## 3. 요청 형식

앱이 `POST /api/generate-after` → RunPod `/run` 으로 보냅니다.

```json
{
  "input": {
    "image_base64": "...",
    "prompt": "...",
    "negative_prompt": "...",
    "strength": 0.35,
    "procedure_id": "alar",
    "purpose": "non_medical_image_simulation"
  }
}
```

응답:

```json
{ "image_base64": "..." }
```

## 4. 품질 업 (선택)

기본은 SDXL img2img입니다. 더 얼굴을 유지하려면 워커에 추가:

- **IP-Adapter / InstantID / FaceID**
- ControlNet (openpose / softedge)
- 자체 LoRA (`training/` 폴더에서 학습한 가중치)

## 5. 비용

Serverless는 **실행 시간(초) × GPU 단가**입니다.  
요청 없으면 거의 0에 가깝습니다. (콜드스타트 시 첫 요청이 느릴 수 있음)

## 6. 고지

의료·시술 권고가 아닙니다. 이미지는 시뮬레이션 목적만 사용하세요.
