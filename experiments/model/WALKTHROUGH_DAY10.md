# Day 10 walkthrough: AWS deploy prep

Day 10 makes the same PoseGuard API ready to go live on AWS, the
way Day 9 made it ready for Render. Nothing about the models
changed. The work is all packaging and paperwork: the repo now
has everything AWS needs to run the API, and the deploy steps are
written down click by click. It also closes the biggest gap on
the resume: AWS showed up in 11 job postings, the only clean gap
in the top 10 from the Nov 9 skills report.

## The pieces, in plain English

Amazon runs your code on its machines, but unlike Render it does
not give you one button. It gives you four separate services that
you wire together, and Day 10 supplies the wiring for all four:

1. **ECR, the image shelf.** ECR (Elastic Container Registry)
   stores the Docker image. `deploy.sh` builds the same slim Day
   9 image and pushes it here. Think of it as a private shelf
   where AWS keeps the exact image it will run.
2. **ECS Fargate, the runner.** ECS runs containers; Fargate is
   the mode where Amazon manages the machines and you only pick
   the size. `deploy/aws/task-definition.json` says: one
   container, `poseguard-api`, 0.25 vCPU and 0.5 GB of memory
   (the smallest Fargate size), port 8000, check it is alive by
   calling `/health`, send logs to CloudWatch. The size is small
   on purpose: the API is a 2 KB model and a web server, it does
   not need more.
3. **Application Load Balancer, the front door.** The internet
   never talks to the container directly. The load balancer
   takes requests on port 80 and forwards them to the container
   on port 8000, and it only sends traffic to containers that
   pass the `/health` check. `deploy/aws/service.json` wires the
   service to it. This is the shape hiring managers expect to
   see, and it is also why the setup costs about $16 a month
   more than the task alone.
4. **S3, the model backup.** `deploy.sh` copies the trained
   model artifacts to an S3 bucket
   (`s3://poseguard-models-<account-id>/models/`) on every
   deploy. The container still serves the models baked into its
   image, exactly as on Day 9; S3 is the versioned backup and
   the handoff point for a retrain. The task role is granted
   read access to that bucket now, so a later slice can teach
   the container to fetch fresh models from S3 without changing
   the permissions.

## Two details worth knowing

**The health check uses Python, not curl.** The slim image is
built from `python:3.12-slim`, which has no curl installed. A
health check that calls curl would fail forever and ECS would
keep killing healthy containers. The task definition instead
runs a one-line Python command that opens
`http://localhost:8000/health` with `urllib`, which is always
available. That command is covered by a test.

**The image URI is a placeholder until deploy time.** The task
definition says
`<account-id>.dkr.ecr.us-east-1.amazonaws.com/poseguard-api:latest`
because the real account ID is his, not something to hardcode in
the repo. `deploy.sh` reads the account ID from the signed-in
AWS CLI, substitutes it with a small Python step (no `jq`
dependency), and registers the rendered copy. The template in
git always keeps the placeholder, and a test checks that it does.

## What was verified, and what was not

Verified here, by tests (`experiments/model/tests/test_aws.py`,
11 checks):

- Both JSON files parse, name the right service, and agree with
  each other: container port 8000 in the task, port 8000 in the
  service load balancer, and `EXPOSE 8000` in the Dockerfile.
- The task is Fargate, smallest size, `awsvpc` networking,
  health check on `/health` via Python, logs to CloudWatch.
- The service wants 1 copy, minimum healthy 100 percent,
  maximum 200 percent, so a redeploy rolls in with no downtime.
- `deploy.sh` contains every step it claims: ECR login, build,
  push, S3 sync, task registration, service create or update.
- The cost note and the honest limits are written in
  `docs/DEPLOY_AWS.md`.

Not verified, and stated honestly in `docs/DEPLOY_AWS.md`:

- `deploy.sh` has never been run end to end. Docker is not
  installed on this machine, and no AWS account is confirmed for
  him. The first real run will surface whatever the tests
  cannot, most likely an IAM permission or a placeholder left
  unfilled in `service.json`, and the fix lands as a later slice.
- There is no live URL yet. The AWS steps need his account and
  his clicks, and the URL only exists after they run.

## What he has to do

About 30 minutes, all in `docs/DEPLOY_AWS.md`: an AWS account
and the AWS CLI signed in, two IAM roles, one ECS cluster, one
load balancer and target group, one security group, then fill in
four placeholders in `service.json` (target group ARN, security
group ID, two subnet IDs) and run `bash deploy/aws/deploy.sh`.
Rough cost once it is live: about $9 a month for the task,
about $16 more for the load balancer, about $25 total. Skipping
the load balancer cuts it to about $9, but then there is no
HTTPS and the address changes on every redeploy.

After the URL exists, it goes into the repo README and the model
README as the live link, next to the Render one. Later deploys
are just re-running `deploy.sh`.

## What to try next

- The camera re-run data is still the fix for the models, not
  the deploy. A live API serving a weak model is still a weak
  model.
- A later slice can teach the container to fetch fresh models
  from S3 at startup (the task role already has read access),
  so a retrain ships without rebuilding the image.
- The React demo shell still needs its own static deploy
  (Vercel); that is separate from this API deploy.
