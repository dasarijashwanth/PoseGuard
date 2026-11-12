# Deploying the PoseGuard API on AWS ECS Fargate (Day 10)

This covers the FastAPI service only (`GET /health`, `POST /predict`),
the same service Day 9 made Render-ready. The React demo shell is a
static site and is deployed separately; that is a later step, not
this one. The Render deploy from Day 9 is untouched by this work.

## What is already done

- `deploy/aws/task-definition.json`: one Fargate task, `poseguard-api`,
  smallest size (0.25 vCPU, 0.5 GB memory), container port 8000,
  health check on `/health`, logs to CloudWatch
  (`/ecs/poseguard-api`). The image is the same slim Day 9 image
  (FastAPI, scikit-learn, pandas, numpy, joblib; no torch, no
  mlflow), pulled from ECR.
- `deploy/aws/service.json`: one ECS service, desired count 1,
  behind an Application Load Balancer target group, in the
  `poseguard` cluster.
- `deploy/aws/deploy.sh`: builds the image, pushes it to ECR, backs
  the trained model artifacts up to S3
  (`s3://poseguard-models-<account-id>/models/`), registers the
  task definition with the real image URI, and creates or updates
  the service.
- The task role is granted read access to that S3 bucket. The
  container still serves the models baked into its image, exactly
  as on Day 9; S3 is the versioned backup and the handoff point
  for a retrain, not something the container reads at startup yet.

## Before you start (his setup, about 30 minutes)

1. An AWS account. If there is none yet, create one at
   https://aws.amazon.com and add a payment method. The first year
   includes 500 MB of ECR storage free; Fargate itself has no
   always-free tier, so the task costs money from day one (see
   costs below).
2. The AWS CLI installed and signed in on the machine that runs
   `deploy.sh` (`aws configure`, region `us-east-1`). The signed-in
   user needs permissions for ECR (push), ECS (register task
   definitions, create and update services), S3 (create bucket,
   put objects), and IAM (pass the two roles below).
3. Docker installed on the same machine. The build VM has no
   Docker, so `deploy.sh` has never been run end to end.
4. Two IAM roles, created once in the AWS Console (IAM > Roles):
   `ecsTaskExecutionRole` (AWS managed policy
   `AmazonECSTaskExecutionRolePolicy`, lets ECS pull the image and
   write logs) and `poseguard-api-task-role` (a custom role with
   read access to `s3://poseguard-models-<account-id>/*`, reserved
   for the retrain flow).

## Console steps (one time)

1. ECS > Clusters > Create cluster: name `poseguard`,
   infrastructure AWS Fargate.
2. EC2 > Load Balancers > Create Application Load Balancer: name
   `poseguard-alb`, scheme internet-facing, listeners HTTP port
   80. Create a target group first (EC2 > Target Groups):
   name `poseguard-api-tg`, type IP, protocol HTTP, port 8000,
   health check path `/health`, VPC default. Note its ARN.
3. EC2 > Security Groups > Create: name `poseguard-api-sg`, VPC
   default, inbound rule HTTP port 80 from anywhere (0.0.0.0/0),
   outbound all traffic. The container port 8000 is reached only
   through the load balancer, never directly from the internet.
   Note the security group ID.
4. VPC: use the default VPC and two of its public subnets. Note
   both subnet IDs.
5. Open `deploy/aws/service.json` and fill in the four
   placeholders: target group ARN, security group ID, and the two
   subnet IDs. Leave everything else as it is.

## Deploy it

1. Push Day 10 to GitHub first (needs his approval; the code is
   not on the repo yet).
2. From the repo root, run:
   `bash deploy/aws/deploy.sh`
   It prints each step as it goes: ECR repo, image push, S3 sync,
   task registration, service create or update. First run takes a
   few minutes, mostly the image build and push.
3. Wait for the service to turn steady (ECS > Clusters >
   `poseguard` > Services > `poseguard-api`, deployments show
   `PRIMARY` running 1/1). Then open the load balancer DNS name
   (EC2 > Load Balancers > `poseguard-alb`, copy the DNS name)
   with `/health` on the end. It should answer with the status
   and both models loaded.
4. Try one prediction (any HTTP client works):
   `POST http://<alb-dns-name>/predict` with a JSON body like
   `{"cue": "head", "samples": [{"elapsed_s": 0.0,
   "head_tilt": -5.622, "shoulder_tilt": 174.173,
   "tracking": 1}]}`, repeated to 10 samples. It answers with a
   state (`upright` or `tilted`) and a probability.
5. Send the live URL back. It goes into the repo README and the
   model README as the live link, next to the Render one.

## Later deploys

Re-run `bash deploy/aws/deploy.sh`. It rebuilds the image, pushes
it, re-syncs the model artifacts to S3, registers a new task
definition revision, and forces a new deployment. The service
rolls the new task in with no downtime (minimum healthy 100
percent, maximum 200 percent). The Render deploy does not change;
the two hosts serve the same image independently.

## Costs (approximate, us-east-1, check the AWS pricing page)

- Fargate task, 0.25 vCPU and 0.5 GB: roughly $9 a month
  always on. There is no free tier for Fargate, so this starts
  the day the service is created.
- Application Load Balancer: roughly $16 a month plus a small
  per-request charge.
- ECR image storage: about $0.10 per GB a month, 500 MB free for
  the first 12 months. The slim image is well under that.
- S3 model artifacts: a few kilobytes, effectively $0.
- Total for the setup as written: roughly $25 a month. The
  cheaper option is to skip the load balancer, give the task a
  public IP, and call the task directly: roughly $9 a month, but
  no HTTPS, and the IP changes on every redeploy, so it is a
  demo shortcut, not a production shape. The guide keeps the load
  balancer because that is the shape hiring managers expect to see.

## What is still unverified

- `deploy.sh` has never been run end to end: no Docker on the
  build VM, and no AWS account is confirmed for him. The files it
  consumes are validated by tests (`experiments/model/tests/
  test_aws.py`): the JSON parses, the image placeholder, port
  8000, the health check, the log config, the service references,
  and the script's key steps. The first real run will surface
  whatever the tests cannot, most likely an IAM permission or a
  placeholder left unfilled, and the fix lands as a later slice.
- The health check in the task definition uses Python's
  `urllib` (`python -c ... urlopen('http://localhost:8000/
  health')`), not curl, because the slim image has no curl. That
  command is covered by a test.
- No live URL exists until the steps above run. Nothing in this
  repo points at a live AWS service yet.

## Resume bullet this closes

"Deployed a containerized FastAPI ML inference service on AWS
(ECS Fargate, ECR, S3) with MLflow experiment tracking." That was
the top skills gap from the Nov 9 report: AWS appeared in 11
postings, the only clean gap in the top 10.
