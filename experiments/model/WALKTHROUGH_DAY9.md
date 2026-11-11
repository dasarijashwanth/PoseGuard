# Day 9 walkthrough: Render deploy prep

Day 9 makes the PoseGuard API ready to go live on Render. Nothing
about the models changed. The work is all packaging: the repo now
has everything Render needs to build the API image and serve it,
and the deploy steps are written down click by click.

## What "deploy-ready" means

A web host like Render runs your code on its machines, not yours.
For that to work it needs three things, and Day 9 supplies all
three:

1. A recipe for the service. `render.yaml` at the repo root is a
   Render Blueprint. It says: one web service, called
   `poseguard-api`, built from the Dockerfile, free plan, and check
   that it is alive by calling `/health`. Render reads this file and
   creates the service in one click.
2. An image that actually builds and starts on Render. The
   Dockerfile had two real blockers, both fixed today.
3. Written steps for the human clicks. `docs/DEPLOY.md` is the
   exact sequence, from opening the Render dashboard to checking
   the live `/health` endpoint.

## The two blockers, in plain English

**Blocker 1: the port.** Render does not run your app on port 8000.
It hands your app a port number through an environment variable
called `$PORT` (usually 10000) and expects your app to listen on
that port. The old Dockerfile hardcoded `--port 8000`, so the app
would have started fine and received no traffic at all. The start
command now reads `$PORT` and falls back to 8000 only when it is
not set, which is the local case.

**Blocker 2: the image was far too fat.** The Dockerfile installed
the eval requirements file, which includes PyTorch and MLflow. The
API never imports either one. Worse, the default PyTorch install is
the multi-gigabyte CUDA build, meant for machines with NVIDIA
graphics cards. On Render that means a slow build and a huge image
for nothing. The Dockerfile now installs a small runtime-only list
(`experiments/model/requirements-api.txt`): FastAPI, uvicorn,
scikit-learn, pandas, numpy, joblib. That is everything the API
imports, and nothing else. The MLflow service in docker-compose
still gets the full requirements through a build arg, so local
MLflow keeps working.

## What was verified, and what was not

Verified live on this machine, by starting the API exactly the way
the Dockerfile will start it (uvicorn, port from `$PORT`):

- `GET /health` answered 200, both models loaded
  (`logistic_regression` for head and shoulder).
- `POST /predict` answered 200 for both cues on a steady upright
  window. Head scored 0.561 tilted and shoulder 0.662, which looks
  wrong for an upright window, and it is the same honest result
  from Day 5: on Run 1 the head model learned the relationship
  backwards, and the shoulder model learned that stillness means
  tilted. The deploy does not fix the models; it serves them.
- A bad cue answered 422, as designed.

Not verified, and stated honestly in `docs/DEPLOY.md`:

- No real Docker build has ever run. Docker is not installed on
  this machine, so the image was checked by tests and by the live
  uvicorn smoke test, not by a build. If the first Render build
  fails, the build log on the Render dashboard is the first place
  to look.
- There is no live URL yet. The Render steps need his clicks and
  his Render account, and the URL only exists after they run.

## What he has to do

About 10 minutes, all in `docs/DEPLOY.md`: push Day 9 to GitHub
first (needs his approval), open the Render dashboard, New >
Blueprint, pick the PoseGuard repo, confirm branch `main`, click
Apply, wait for the build, then open the live URL with `/health`
on the end and confirm it answers. No API keys, no env vars, no
secrets to set. The free plan sleeps after 15 minutes of no
traffic, so the first request after a sleep takes about 30 to 60
seconds; that is normal.

After the URL exists, it goes into the repo README and the model
README as the live link, and the next deploy happens automatically
on every push to `main`.

## What to try next

- Camera re-run data is still the fix for the models, not the
  deploy. A live API serving a weak model is still a weak model.
- The AWS deploy (EC2, ECR, S3) is still open as a later day; it
  closes the AWS gap on the resume and reuses the same slim image.
- The React demo shell still needs its own static deploy (Vercel);
  that is separate from this API deploy.
