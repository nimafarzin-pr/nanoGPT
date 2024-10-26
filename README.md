## Here’s a step-by-step guide on how to train your own NanoGPT model on a Mac, based on your dataset.

### Prerequisites

Ensure you have the following installed:

1. **Python 3.x**: Make sure you have a recent version of Python installed.
2. **pip**: Python package installer.
3. **PyTorch**: Recommended installation with support for Metal Performance Shaders for Apple Silicon.

## Option 1: Using a Virtual Environment

### Step 1: **Clone the Repository**

```bash
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
```

### Step 2: **Set Up the Environment**

- Create a virtual environment and install dependencies:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- For M1, install PyTorch with MPS:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

### Step 3: **Prepare the Dataset**

- Place your dataset (e.g., `data.txt`) in `data/`.
- Create a dataset directory:
  ```bash
  mkdir data/my_dataset
  ```
- Use a preprocessing script to convert the dataset into a format suitable for training:
  ```bash
  python data/shakespeare_char/prepare.py --input_file=data/data.txt --output_dir=data/my_dataset
  ```
- Update `prepare.py` for specific tokenization needs.

### Step 4: **Configure the Training Settings**

- Edit configurations in `config/` or create a custom configuration file.
- Set hyperparameters based on your dataset size and available memory (e.g., batch size, learning rate).

### Step 5: **Train the Model**

- Start training with the MPS device for M1 GPUs:
  ```bash
  python train.py --data_dir=data/my_dataset --device=mps
  ```
- Training can be adjusted by modifying batch size or model parameters if out-of-memory errors occur.

### Step 6: **Monitor Training with TensorBoard**

- Install TensorBoard:
  ```bash
  pip install tensorboard
  tensorboard --logdir=out
  ```
- Access TensorBoard via `http://localhost:6006/` to visualize training progress.

### Step 7: **Generate Text Samples**

- Generate text with the trained model:
  ```bash
  python sample.py --out_dir=out --start="Your initial text here"
  ```
- Tune parameters like temperature and max length for varied results.

### Step 8: **Fine-Tuning Tips**

- Start with a small dataset for initial runs.
- Experiment with different hyperparameters.
- Pre-train on a larger corpus if the dataset is small to improve performance.

### Step 9: Deactivate the Virtual Environment (When Done)

Deactivate the virtual environment when you are finished:

```bash
deactivate
```

---

## Option 2: Without Virtual Environment

### Step 1: Install Required Packages

Start by installing the dependencies. Open a terminal and run:

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

These libraries are needed for training, tokenization, dataset handling, logging, and progress tracking.

### Step 2: Set Up the NanoGPT Repository

1. **Clone the NanoGPT Repository**:

   ```bash
   git clone https://github.com/karpathy/nanoGPT.git
   cd nanoGPT
   ```

2. **Prepare Your Dataset**:(in our case `"ufo"`)

   - Organize your dataset in a text format. For example, if you're training on a custom text file, save it as `data/your_dataset/your_text.txt` or `data/ufo/UFOs_coord.csv in our case` for example https://www.kaggle.com/datasets/utkarshx27/ufo-sights-2016-us-and-canada.

3. **Preprocess the Data**:

   - Create a script similar to `data/shakespeare_char/prepare.py`, which tokenizes and converts your text into binary files for training and validation (`train.bin` and `val.bin`).
   - Example preprocessing script (`data/your_dataset/prepare.py`):

     ```python
     import os
     from tiktoken import Encoding

     # Load your dataset csv or txt
     #in our case "data/ufo/UFOs_coord.csv"
     df = pd.read_csv('data/ufo/UFOs_coord.csv',encoding='ISO-8859-1')
     data= df['Summary'].str.cat(sep='\n')

     #or something like below two line
     with open('data/your_dataset/your_text.txt', 'r') as f:
         data = f.read()

     # Tokenize the text
     encoder = Encoding.get_encoder('gpt2')
     tokens = encoder.encode(data)

     # Split tokens into training and validation
     split_idx = int(len(tokens) * 0.9)
     train_tokens = tokens[:split_idx]
     val_tokens = tokens[split_idx:]

     # Save binary files
     os.makedirs('data/your_dataset', exist_ok=True)
     train_tokens.tofile('data/your_dataset/train.bin')
     val_tokens.tofile('data/your_dataset/val.bin')
     ```

4. **Run the Preprocessing Script**:
   ```bash
   python data/your_dataset/prepare.py
   ```

### Step 3: Configure the Training Parameters(in our case `"train_on_ufo"`)

1. **Create a Training Configuration File**:

   - Start with `config/train_shakespeare_char.py` as a template.
   - Adjust parameters based on your dataset and computational resources:

     ```python
     # config/train_your_dataset.py
     #in our case "out-ufo" you can find our case in config/train_on_ufo.py
     out_dir = 'out-your-dataset'
     eval_interval = 500
     eval_iters = 200
     log_interval = 100

     # Training settings
     batch_size = 64
     block_size = 256  # Context size
     n_layer = 6
     n_head = 6
     n_embd = 384
     dropout = 0.1
     learning_rate = 3e-4
     max_iters = 10000
     lr_decay_iters = 10000
     min_lr = 6e-5
     beta2 = 0.99
     warmup_iters = 100

     # Model loading/saving
     init_from = 'scratch'  # Initialize a new model
     ```

### Step 4: Train the Model

1. **Run the Training Script**:
   (`"At least 500 iteration"`)

   - If using an Apple Silicon Mac, use the MPS backend for better performance:
     ```bash
     python train.py config/train_your_dataset.py --device=mps
     ```
   - If using a standard Mac with an Intel CPU:
     ```bash
     python train.py config/train_your_dataset.py --device=cpu --compile=False
     ```

2. **Monitor Training**:
   - Optionally, you can use Weights and Biases (`wandb`) for logging and monitoring:
     ```bash
     wandb login
     ```
   - Then set `wandb_project = 'your_project_name'` in your config file.

### Step 5: Finetuning (Optional)

If you want to finetune a pretrained model, update your configuration to initialize from a GPT-2 checkpoint (e.g., `init_from='gpt2'`), and adjust the learning rate to a smaller value (e.g., `1e-5`).

### Step 6: Generate Samples

1. **Run the Sampling Script**:

   ```bash
   python sample.py --out_dir=out-your-dataset --start="Your prompt here" --device=mps
   ```

2. **Adjust Sampling Parameters**:
   - You can control the number of generated tokens, sampling temperature, etc., in the script.

### Step 7: Troubleshooting and Optimization

1. **Disable PyTorch 2.0 Compile Mode**:
   If you encounter issues with PyTorch's compile mode, run training with `--compile=False`.

2. **Tune Hyperparameters**:
   - Adjust the number of layers, heads, and embedding dimensions.
   - Experiment with different learning rates and dropout values.

### Step 8: Further Fine-Tuning and Evaluation

Evaluate your model's performance on tasks like text generation, and fine-tune it based on results. Adjust hyperparameters as needed to improve quality.

This guide should help you train your own NanoGPT model on a Mac with ease!
