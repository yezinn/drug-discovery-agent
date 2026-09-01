# Kubernetes 배포를 염두에 둔 최소 컨테이너 이미지.
# 실제 K8s 매니페스트(Deployment/Service)까지는 이번 프로젝트 범위 밖으로 두고,
# "컨테이너화까지는 직접 해봤다"는 근거로 Dockerfile만 준비함 — 면접에서
# "다음 단계로 K8s 배포를 어떻게 하시겠어요?"에 답할 수 있는 정도로 충분하다고 판단.

FROM python:3.11-slim

WORKDIR /app

# RDKit 등 일부 패키지가 필요로 하는 최소 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
