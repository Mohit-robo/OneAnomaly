# OneAnomaly: End-to-End Deployment Guide

This document outlines the detailed step-by-step process for deploying the OneAnomaly application into production using our hybrid Cloud-Edge architecture. 

## Phase 1: Edge GPU Gateway Provisioning (Local Site)
This phase configures the local GPU machine that will execute heavy tensor operations via Triton.

1. **Pre-requisites:**
   * Ensure Docker and NVIDIA Container Toolkit (`nvidia-docker2`) are installed.
   * Ensure the `models/` directory contains the required DINOv3 `.pth` weights.
   * Run the ONNX/TRT export scripts locally (these run strictly offline) to populate the `triton_models/dinov3_onnx/1/` directory.

2. **Tailscale Setup (Edge Node):**
   * Install Tailscale on the local GPU machine: `curl -fsSL https://tailscale.com/install.sh | sh`
   * Authenticate and bring the node online: `sudo tailscale up`
   * Note the Tailscale IP address of this machine (e.g., `100.x.y.z`).

3. **Start the Edge Stack:**
   * Review `docker-compose.local.yml` and ensure the `.env` variables match your paths.
   * Start the gateway and Triton server:
     ```bash
     docker compose -f docker-compose.local.yml up -d --build
     ```
   * Verify health: `curl http://localhost:8080/health` (Should return `"triton_connected": true`).

---

## Phase 2: AWS Infrastructure Provisioning (Cloud Site)
This phase sets up the cloud tier that handles web traffic, API routing, and S3 configuration persistence.

### A. Amazon S3 (State Management)
1. Navigate to the **AWS S3 Console**.
2. Create a new bucket named `oneanomaly-sessions-<your-org-name>`.
3. Uncheck "Block all public access" **ONLY IF** you plan to serve images directly from S3 to the frontend in the future. Otherwise, leave it blocked for better security.
4. Note the exact bucket name for later.

### B. Deploying the Cloud API Server (EC2)
To completely bypass AWS App Runner costs and limitations, we will host the Cloud API directly on a lightweight EC2 instance connected to your Tailscale network.

1. **Launch a Cloud Instance:**
   * Launch a free-tier **t3.micro EC2 Instance** in AWS (Ubuntu 24.04).
   * Note its Public IP Address (e.g., `13.x.y.z`).
   * Connect to the instance via SSH.

2. **Tailscale Setup (Cloud Node):**
   * Install Tailscale: `curl -fsSL https://tailscale.com/install.sh | sh`
   * Bring the node online: `sudo tailscale up`

3. **Install Dependencies:**
   * Install Docker on the EC2 instance:
     ```bash
     sudo apt-get update
     sudo apt-get install docker.io docker-compose-v2 git -y
     sudo usermod -aG docker ubuntu
     ```
   * Log out and log back in (or `newgrp docker`) for the group change to take effect.

4. **Deploy the Cloud API:**
   * Clone your code repository onto the EC2 instance:
     ```bash
     git clone https://github.com/<your-username>/anomaly_app.git
     cd anomaly_app
     ```
   * Ensure `docker-compose.cloud.yml` has the correct `GATEWAY_URL` pointing to the **Tailscale IP** of your Local Edge Node (e.g., `http://100.x.y.z:8080`).
   * Start the Cloud container:
     ```bash
     docker compose -f docker-compose.cloud.yml up -d --build
     ```

---

## Phase 3: Final Verification
1. **Access the Application:** Open your browser and navigate to the Public IP of your EC2 instance on port 8080: `http://<EC2-Public-IP>:8080`.
2. **Test Upload & Preprocessing:** Upload an image. The preprocessing preview should load immediately (happens on the EC2 instance).
3. **Test Feature Extraction:** Start a feature extraction job. The Cloud API will proxy this request directly to the Edge Gateway via Tailscale. You should see the Gateway successfully processing crops via Triton.
4. **Test S3 Persistence:** Click "Save Session Config". Verify in the AWS S3 console that a `.json` file appears in the `oneanomaly-sessions` bucket.

### Troubleshooting
* **Gateway Unreachable (503):** The Cloud API cannot reach the Edge node. Ensure Tailscale is active on both machines (`tailscale status`) and the `GATEWAY_URL` in `docker-compose.cloud.yml` accurately matches your local GPU node's Tailscale IP.
* **S3 Access Denied:** Ensure your EC2 Instance Profile has read/write permissions to the `oneanomaly-sessions` S3 bucket, or that you passed the correct `AWS_ACCESS_KEY_ID` into the `.env` if not using IAM roles.
* **Site Cannot Be Reached:** Ensure your EC2 Security Group allows inbound TCP traffic on port `8080` from the Internet (`0.0.0.0/0`).
