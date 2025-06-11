# ShorterBetter

**Paper**: [ShorterBetter: Guiding Reasoning Models to Find Optimal Inference Length for Efficient Reasoning](https://arxiv.org/abs/2504.21370)

**Model**: 

SB-1.5B: [🤗 Hugging Face Model](https://huggingface.co/Justin6657/SB_DS1.5B_alpha_2)

SB-7B: [🤗 Hugging Face Model](https://huggingface.co/JingyangYi/SB_DS7B_alpha_2/tree/main)

ShorterBetter is a project for training and enhancing language models to generate shorter, better responses using the VERL (Versatile Efficient Reinforcement Learning) framework.

## Installation

### Step 1: Install VERL Framework

1. **Clone the repository with VERL submodule**:
   ```bash
   git clone --recursive https://github.com/your-username/ShorterBetter.git
   cd ShorterBetter
   ```

2. **Install VERL dependencies**:
   
   Follow the [official VERL installation guide](https://verl.readthedocs.io/en/latest/start/install.html) for detailed instructions. The basic installation involves:
   
   ```bash
   # Install from source
   cd verl
   pip install -e .
   
   # Install additional dependencies for training backends
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install flash-attn --no-build-isolation
   pip install vllm>=0.8.0  # For rollout generation
   ```

### Step 2: Install ShorterBetter Dependencies

3. **Install additional ShorterBetter dependencies**:
   ```bash
   cd ..  # Back to ShorterBetter root
   pip install -r requirements.txt
   ```

## Data Structure

### Training Data
Training datasets are prepared and located in:
- **Location**: `/ShorterBetter/deepscaler/data/`

### Evaluation Data
Evaluation datasets are prepared and located in:
- **Location**: `/ShorterBetter/eval_data/`
- **Math evaluation**: `/ShorterBetter/eval_data/math/`
- **Out-of-distribution evaluation**: `/ShorterBetter/eval_data/ood/`

## Training

### Available Training Scripts

The training scripts are located in `scripts/train/` and include:

1. **`sb_7b.sh`** - Training script for 7B parameter models
2. **`sb_1.5B.sh`** - Training script for 1.5B parameter models

### Running Training

1. **Configure your environment variables**

2. **Customize reward function** (optional):
   - Edit `/ShorterBetter/verl/verl/workers/reward_manager/naive.py` line 244
   - Adjust `alpha` and `beta` parameters (default: alpha=2.0, beta=0.001)

3. **Run training**:
   ```bash
   # For 7B model
   bash scripts/train/sb_7b.sh 
   
   # For 1.5B model  
   bash scripts/train/sb_1.5B.sh
   ```


### Monitoring Training

The training process outputs accuracy and length statistics for each batch. Use the provided analysis script:

```bash
python scripts/train/check_acc_len.py
```

## Evaluation

### Math Evaluation

1. **Generate responses**:
   ```bash
   bash scripts/eval/math/math_eval.sh
   ```

2. **Verify results**:
   ```bash
   bash scripts/eval/math/verifier.sh
   ```
   
   Or use the Python script directly:
   ```bash
   python scripts/eval/math/verifier.py --dataset_dir /path/to/outputs --output_dir /path/to/verified --batch_size 16
   ```

### Out-of-Distribution (OOD) Evaluation

1. **Generate responses**:
   ```bash
   bash scripts/eval/ood/ood_eval.sh
   ```

2. **Verify results**:
   ```bash
   # For BBH (Big Bench Hard) evaluation
   python scripts/eval/ood/bbh_verify.py
   
   # For general OOD evaluation
   python scripts/eval/ood/ood_eval.py
   ```
### Coding Tasks Evaluation


```bash
bash ./code_eval/example_scripts/eval_code_example.sh
```

## Results and Outputs

### Sample Outputs
To see example outputs from trained models and baseline comparisons:
- **Location**: `/ShorterBetter/eval_data/outputs/`
- **Structure**:
  - `sample/` - Sample outputs for quick inspection
  - `math/` - Math evaluation outputs
  - `ood/` - Out-of-distribution evaluation outputs

