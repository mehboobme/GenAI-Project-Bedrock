# End to End Gen AI Project using AWS Bedrock

## How to run ?

```bash
conda create -n bedrockproject python=3.8 -y
```

```bash
conda activate bedrockproject
```

```
pip install -r requirements.txt
```

### Install aws cli from the following link:
```bash
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
```

### Add credentials by running the following command
```bash
aws configure
```

```bash
streamlit run research/bedrock_trials.py
```

```bash
streamlit run main.py
```