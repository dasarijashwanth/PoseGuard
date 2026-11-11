# Deploying the PoseGuard API on Render (Day 9)

This covers the FastAPI service only (`GET /health`, `POST /predict`).
The React demo shell is a static site and is deployed separately;
that is a later step, not this one.

## What is already done

- `render.yaml` at the repo root is a Render Blueprint. It defines one
  web service, `poseguard-api`, Docker runtime, free plan, health
  check on `/health`. Render reads this file and creates the service
  in one click.
- The Dockerfile is Render-ready: it installs only the small
  runtime-only requirements (`experiments/model/requirements-api.txt`),
  and its start command binds `$PORT`, which Render sets (10000 by
  default, not 8000).
- The image already contains the trained models, so it serves
  immediately. No API keys or env vars are needed. There is nothing
  secret to set on Render.

## Deploy it (his clicks, about 10 minutes)

1. Push Day 9 to GitHub first (needs his approval; the code is not
   on the repo yet).
2. Go to https://dashboard.render.com and sign in. If there is no
   account yet, create one with "Sign up with GitHub", it is the
   fastest route and it connects the repo in the same step.
3. Click **New +** (top right), then **Blueprint**.
4. Pick the **PoseGuard** repo from the list. If GitHub is not
   connected yet, Render asks to connect it; approve the
   `dasarijashwanth/PoseGuard` repo only.
5. Make sure the branch is **main**. Render shows one service from
   `render.yaml`: name `poseguard-api`, Runtime Docker, Plan Free.
6. Click **Apply**. Render starts the first build, which takes a
   few minutes. The first build is the slow one; later deploys reuse
   the cached layers.
7. When the service status turns **Live**, Render shows a URL like
   `https://poseguard-api.onrender.com`. Open it with `/health` on
   the end and confirm it answers, for example:
   `https://poseguard-api.onrender.com/health`
   should return `{"status": "ok", "models": ["head", "shoulder"], ...}`.
8. Try one prediction (any HTTP client works):
   `POST https://poseguard-api.onrender.com/predict` with a JSON body
   like `{"cue": "head", "samples": [{"elapsed_s": 0.0,
   "head_tilt": -5.622, "shoulder_tilt": 174.173, "tracking": 1}]}`,
   repeated to 10 samples. It answers with a state (`upright` or
   `tilted`) and a probability.
9. Send the live URL back. It goes into the repo README and the
   model README as the live link.

## Free tier notes

- A free web service sleeps after 15 minutes with no traffic. The
  first request after a sleep takes about 30 to 60 seconds while it
  wakes up; that is normal, not an error.
- Render rebuilds and redeploys automatically on every push to
  `main`, so later day-slices ship themselves once pushed.
- The health check on `/health` is how Render knows a deploy
  worked. If a deploy fails the health check, Render keeps the last
  good version live.

## If the Blueprint route fails

Manual fallback: Dashboard > **New +** > **Web Service** > pick the
PoseGuard repo > Runtime **Docker** > Dockerfile Path
`experiments/model/Dockerfile` > Health Check Path `/health` >
Instance Type **Free** > **Create Web Service**. Same result, just
more clicks.

## What is still unverified

- No real Docker build has ever run: Docker is not installed on the
  build machine, so the image was checked by tests and by running
  the same start command locally with uvicorn, not by a real build.
  If the first Render build fails, the build log on the Render
  dashboard is the first place to look, and the fix lands as a
  Day 10 slice.
- The live URL does not exist until the steps above run. Nothing in
  this repo points at a live service yet.
