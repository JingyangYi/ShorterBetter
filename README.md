# ShorterBetter

ShorterBetter is a project for training and enhancing language models to generate shorter, better responses.

## Setup

### Environment Variables

Before running the scripts, set up the following environment variables:

```bash
# Core project paths
export PROJECT_HOME="/path/to/your/project"  # Root directory of the project
export DATASET_DIR="/path/to/your/datasets"  # Directory containing training datasets
export LOG_DIR="/path/to/your/logs"          # Directory for storing logs

# API keys
export WANDB_API_KEY="your_wandb_api_key"    # Your Weights & Biases API key
```

### Using Script Templates

The `scripts` directory contains template files that need to be customized with your own paths and settings:

1. Create your personal version of a script:
   ```bash
   cp scripts/sb_dsqwen1.5B_template.sh scripts/sb_dsqwen1.5B_personal.sh
   ```

2. Edit the personal script to use your own paths and settings:
   ```bash
   vim scripts/sb_dsqwen1.5B_personal.sh
   ```

3. Run your personal script:
   ```bash
   bash scripts/sb_dsqwen1.5B_personal.sh
   ```

## Analyzing Results

Use the `check_acc_len_template.py` script to analyze training results:

```bash
python scripts/check_acc_len_template.py --log_file /path/to/your/output.log --prompts_per_step 64 --responses_per_prompt 8 --max_steps 500
```

## Notes

- Personal script versions with suffix `_personal.sh` or `_local.sh` are added to `.gitignore` and won't be committed to the repository.
- Large model files (*.pt), checkpoints, and logs are also excluded from version control. # ShorterBetter
