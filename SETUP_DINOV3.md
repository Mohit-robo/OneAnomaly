# DINOv3 Model Setup Instructions

## Important: Model Weights Download

DINOv3 model weights are **not publicly downloadable** via direct URLs. You need to request access from Meta.

### Steps to Download Model Weights

1. **Request Access**
   - Visit: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
   - Fill out the form with your information
   - You'll receive an email with download links

2. **Download the Model**
   - For this project, we're using **ViT-S/16** model
   - Download: `dinov3_vits16_pretrain_lvd1689m.pth`
   - The email will contain a temporary download link

3. **Place the Model File**
   - Create a `models` directory in the project root:
     ```
     d:\Mohit\anomaly_app\models\
     ```
   - Place the downloaded `.pth` file there:
     ```
     d:\Mohit\anomaly_app\models\dinov3_vits16_pretrain_lvd1689m.pth
     ```

### Alternative: Use wget with the Link from Email

Once you receive the email, you can use wget to download:

```bash
# Example (replace URL with the one from your email)
wget -O models/dinov3_vits16_pretrain_lvd1689m.pth "YOUR_DOWNLOAD_LINK_FROM_EMAIL"
```

### Current Project Structure

```
anomaly_app/
├── python/
│   ├── dinov3/              # DINOv3 repository (cloned)
│   ├── test_dinov3_pytorch.py
│   ├── feature_extractor.py
│   ├── memory_bank.py
│   ├── anomaly_detector.py
│   ├── api_server.py
│   └── requirements.txt
├── models/                  # Create this directory
│   └── dinov3_vits16_pretrain_lvd1689m.pth  # Place downloaded weights here
├── Data/                    # Your test images
├── server.js
└── package.json
```

### Testing After Setup

Once you've placed the model weights, test with:

```bash
cd python
python test_dinov3_pytorch.py --input_folder ../Data --output_folder results --model dinov3_vits16
```

### Troubleshooting

**Error: HTTP 403 Forbidden**
- This means the direct download is blocked
- You MUST download manually after requesting access

**Error: Model file not found**
- Make sure the file is in `d:\Mohit\anomaly_app\models\`
- Check the filename matches exactly

**Error: CUDA out of memory**
- Use `--device cpu` flag for testing
- Or reduce the number of images with `--max_images 5`
