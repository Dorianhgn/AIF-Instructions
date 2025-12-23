# Part 1: Building and Training the "Quick, Draw\!" Model

Welcome to the first part of our project\! The goal here is to build the "brain" of our application. We'll train a Convolutional Neural Network (ConvNet) to recognize 10 classes of drawings from the "Quick, Draw\!" dataset.

We will build everything from the ground up: setting up the environment, exploring the data, building and training a PyTorch model, and finally professionalizing our script with argument parsing and TensorBoard logging.

Let's get started.

## Step 0: Git Repository Initialization

Before writing any code, we need to set up version control. Git allows you to track changes, collaborate with others, and maintain a clean project history.

### 0.1: Initialize Your Repository

1. **Create a new repository on GitHub or GitLab**:
   - Go to GitHub and create a new repository
   - Name it something meaningful like `Quick_Draw_Vision_Project`
   - Do **not** initialize it with a README yet (we'll do that locally)

2. **Initialize Git locally**:
   ```bash
   git init
   ```

3. **Create a README**:
   Create a `README.md` file with your project title:
   ```markdown
   # Quick, Draw! Vision Project
   
   A deep learning project to recognize hand-drawn sketches using a Convolutional Neural Network.
   ```

4. **Add a Python `.gitignore`**:
   - Download the official Python `.gitignore` from [GitHub's gitignore repository](https://github.com/github/gitignore/blob/main/Python.gitignore)
   - Save it as `.gitignore` in your project root
   - This prevents Python cache files, virtual environments, and other unnecessary files from being tracked

5. **Make your first commit**:
   ```bash
   git add README.md .gitignore
   git commit -m "Initial commit: Add README and .gitignore"
   ```

6. **Push to remote**:
   ```bash
   git branch -M main
   git remote add origin <your-repository-url>
   git push -u origin main
   ```

### 0.2: Create a Development Branch

We will use **feature branches** to keep our `main` branch clean and stable.

1. **Create a new branch** for backend development:
   ```bash
   git checkout -b dev-backend
   ```

2. **Why use branches?**
   - `main` stays clean and deployable
   - `dev-backend` is where you experiment and build
   - Later, you'll merge via Pull Request for review

**Important:** From now on, you should make **clear and informative commits** at each important step of the project. Good commit messages help you and others understand what changed and why.

## Step 1: Environment Setup

First, we need to set up a clean, reproducible environment.

1.  **Create `requirements.txt`**: Create a file named `requirements.txt` and add the following dependencies. These are the only packages that will be installed inside our final Docker container.

    ```
    torch==2.0.1
    torchvision==0.15.2
    tensorboard==2.14.0
    pandas==2.0.3
    numpy==1.24.4
    matplotlib==3.7.2
    ```

2.  **Create Virtual Environment**: It's crucial to use a virtual environment. We recommend using a specific Python version (like 3.10) because our Docker container will also use a specific version. This avoids "it works on my machine" problems.

    ```bash
    # Example using python 3.10
    python3.10 -m venv .venv
    ```

3.  **Activate Environment**:

      * On macOS/Linux (or WSL): `source .venv/bin/activate`
      * On Windows: `.\.venv\Scripts\activate`

4.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

5.  **Install Jupyter**: We'll use Jupyter for exploration. We install it *outside* the `requirements.txt` to keep our production container light.

    ```bash
    pip install jupyter notebook
    ```

6. **Commit your changes**:
   ```bash
   git add requirements.txt
   git commit -m "Add project dependencies"
   ```

## Step 2: Get the Data

We will use the "Quick, Draw\!" dataset.

1.  Go to the dataset source: <a href="https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn" target="_blank" rel="noopener noreferrer">https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn</a>
2.  Download the `.npy` files for the following 10 classes:
      * `cat`
      * `dolphin`
      * `moon`
      * `alarm clock` (or `alarm`)
      * `banana`
      * `airplane`
      * `circle`
      * `door`
      * `donut`
      * `eye`
3.  Create a `data/` directory in your project root and move all 10 `.npy` files into it.
4.  **Update `.gitignore`**: Add the following line to your `.gitignore` to avoid committing large data files:
   ```
   data/
   ```
5.  **Commit your progress**:
   ```bash
   git add .gitignore
   git commit -m "Configure gitignore to exclude data files"
   ```

## Step 3: Data Exploration (EDA)

Let's see what this data looks like.

1.  Create a new notebook (eg. `EDA.ipynb`).
2.  **Task**: In this notebook, load one of the `.npy` files.
    <details>
    <summary>🕵️‍♂️ Hint</summary>
    Use `numpy.load()`.
    </details>
4.  **Task**: What is the shape of the data?
5.  **Task**: Visualize a few images with `plt.imshow`.

## Step 4: Build the Model (`model.py`)

Now, let's define our network architecture.

1.  Create a new file named `model.py`.

2.  **Task**: Add the following skeleton code. You must fill in the `...` parts.

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvNet(nn.Module):
        def __init__(self, dropout_rate=0.5):
            super(ConvNet, self).__init__()
            # TODO: Define all your layers here
            # self.conv1 = ...
            # self.pool1 = ...
            # ...
            # self.dropout = nn.Dropout(dropout_rate)
            # ...

        def forward(self, x):
            # TODO: Define the forward pass
            # x = self.pool1(F.relu(self.conv1(x)))
            # ...
            # x = torch.flatten(x, 1) # Flatten tensor for the linear layer
            # ...
            return x
        
        def get_features(self, x):
            # This method will be used for TensorBoard embeddings
            # It should return the flattened output of the last conv layer
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            return x
    ```

3.  **Instructions**:

      * The `__init__()` method instantiates the layers.
      * The `forward()` method describes the data flow.
      * **Architecture**: Implement the following architecture:
        1.  `Conv2d` (16 filters, kernel size 3) -\> `ReLU` -\> `MaxPool2d` (kernel size 2, stride 2)
        2.  `Conv2d` (32 filters, kernel size 3) -\> `ReLU` -\> `MaxPool2d` (kernel size 2, stride 2)
        3.  `Dropout` (use the `dropout_rate` parameter)
        4.  `Flatten` (we've given you this one)
        5.  `Linear` (128 output features) -\> `ReLU`
        6.  `Linear` (10 output features, for our 10 classes)
      * **Challenge**: You must **figure out the `in_channels` and `out_channels`** for each `Conv2d` layer, and the `in_features` for the first `Linear` layer. Do not use `padding` or `dilation`. Check the PyTorch documentation\!

4. **Commit your model**:
   ```bash
   git add model.py
   git commit -m "Implement ConvNet architecture"
   ```

## Step 5: Create the Training Logic (`train.py`)

Let's create the script that will actually run the training.

1.  Create a new file `train.py`.

2.  **Task**: Add the following function skeletons. You will fill them in.

      * `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
      * We've provided the `train` function you'll be using.

    <!-- end list -->

    ```python
    import torch
    import torch.nn as nn
    from tqdm import tqdm
    from statistics import mean

    import os
    from torch.utils.tensorboard import SummaryWriter
    import torchvision

    # --- THIS IS THE TRAIN FUNCTION ---
    # (We will add val_loader and writer later)
    def train(net, optimizer, train_loader, device, epochs=10):
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            running_loss = []
            t = tqdm(train_loader)
            for x, y in t:
                x, y = x.to(device), y.to(device) # Move data to device
                
                # TODO: Forward pass
                outputs = ...
                
                # TODO: Calculate loss
                loss = ...
                
                running_loss.append(loss.item())
                
                # TODO: Backward pass and optimization
                optimizer.zero_grad()
                ... # backward loss
                ... # step of the optimizer
                
                t.set_description(f'training loss: {mean(running_loss)}')
        return running_loss

    # --- THIS IS THE TEST FUNCTION SKELETON ---
    def test(model, test_loader, device):
        model.eval() # Set model to evaluation mode
        test_corrects = 0
        total = 0
        with torch.no_grad(): # Disable gradient calculation
            for x, y in test_loader:
                # TODO: Move data to device
                x, y = ...
                
                # TODO: Get model predictions
                y_hat = ...
                
                # TODO: Get the class with the highest score (argmax)
                predictions = ...
                
                # TODO: Count correct predictions
                test_corrects += ...
                total += y.size(0)
                
        return test_corrects / total

    # We will add the main execution block later
    # if __name__ == '__main__':
    #     ...
    ```

## Step 6: First Training (In your notebook)

Let's test our model and functions.

1.  Go back to your `EDA.ipynb` notebook.
2.  **Task**: Import your new modules:
    ```python
    from model import ConvNet
    from train import train, test
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    ```
3.  **Task: Data Prep**
      * Load all 10 `.npy` files.
      * Combine them into one giant `X` array and one `y` array for labels.
      * **Split your data** into train, validation, and test sets. (e.g., 70% train, 15% val, 15% test).
      * Convert them to PyTorch Tensors.
      * Create `TensorDataset` objects for each split.
4.  **Task: Create `DataLoader`s** for your train, val, and test `TensorDataset`s, with a **careful choice** of this parameters.
      * **`batch_size`**: This is a trade-off. A small size is inefficient; a large size can cause an OOM (Out Of Memory) error. You can try different values and watch your computer's memory\!
      * **`shuffle`**: `True` for ??? (to randomize data) and `False` for ???.
      * **`num_workers`**: This uses parallel processes to load data. A good rule of thumb is ~20% of your CPU threads.
        <details>
        <summary>🕵️‍♂️ Hint: How to find CPU threads</summary>
        On WSL/Linux, run `nproc`. On Windows, open Task Manager, go to the Performance tab, and look at "Logical processors".
        </details>
      * **`random_state`** : To fix your splits for reproductible runs.
5.  **Task: Train\!**
      * Set your device: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
      * Initialize your model: `net = ConvNet(dropout_rate=0.5).to(device)`
      * Define an optimizer: `optimizer = optim.SGD(net.parameters(), lr=0.001)` (Try `SGD` or `AdamW` too\!)
      * Run the training: `losses = train(net, optimizer, train_loader, device, epochs=5)`
      * Plot the `losses` with `matplotlib`. Did it learn?
      * Run testing: `acc = test(net, test_loader, device)`

## Step 7: Add the validation pass in the training loop

Now that we have a working training loop, let's add validation to monitor overfitting.
1.  **Task**: Modify the `train` function in `train.py` to accept a `val_loader` and `train_loader` parameter.
2.  Inside the epoch loop, after the training pass, add a validation pass:

    ```python
    # After training loop in each epoch
    val_loss = []
    val_corrects = 0
    total = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for x_val, y_val in val_loader:
            # TODO : Move data to device
            x_val, y_val = ...

            # TODO : Get model predictions
            outputs_val = ...

            # TODO : Calculate loss
            loss_val = ...
            val_loss.append(loss_val.item())

            # TODO : Get the class with the highest score (argmax)
            predictions_val = ...

            # TODO : Count correct predictions
            val_corrects += ...
            total += y_val.size(0)

    val_acc = val_corrects / total
    ```

3.  **Task**: Modify your `tqdm` description to show both train and val loss.
    ```python
    t.set_description(f'training loss: {mean(running_loss):.4f}, val loss: {mean(val_loss):.4f}, val acc: {val_acc:.4f}')
    ```

4. **Test your changes**: Go back to your notebook and re-run the training with the modified `train` function, passing in the `val_loader`. Check that validation loss and accuracy are printed each epoch. Plot the training and validation losses to see how they evolve. Did the lmodel overfit?

## Step 8: Professionalizing the `train.py` script

That notebook was great for testing, but real projects use runnable scripts. Let's move all that logic into `train.py`.

You have to fill this block:

```python
if __name__ == '__main__':
    # 1. Parse command-line arguments with argparse (see below)

    if not os.path.exists('runs'):
        os.makedirs('runs')
    writer = SummaryWriter(f'runs/{exp_name}')

    # 2. Load and preprocess data (sklearn train_test_split, TensorDataset)
    # 3. Create DataLoaders
    # 4. Initialize model, optimizer with parsed args. I recommend to use AdamW.
    # 5. Call train() with train_loader and val_loader
    # 6. Test the model on test_loader and print accuracy
    # 7. Save the model weights to weights/{exp_name}_net.pth
    if not os.path.exists('weights'):
        os.makedirs('weights')
    torch.save(net.state_dict(), f'weights/{exp_name}_net.pth')

    # 8. Add TensorBoard logging and embedding visualization here (No edit needed)
    print("Adding embeddings to TensorBoard...")
    #8.a) Get 256 random images and labels from your train_dataset
    perm = torch.randperm(len(train_dataset)) 
    images, labels = train_dataset.tensors[0][perm][:256], train_dataset.tensors[1][perm][:256]
    images = images.to(device)

    # 8.b) Get embeddings from the model
    with torch.no_grad():
        embeddings = net.get_features(images) # Use the method you defined!

    # 8.c) Add to TensorBoard
    writer.add_embedding(embeddings,
                        metadata=labels,
                        label_img=images.reshape(-1, 1, 28, 28), # Reshape for TB
                        global_step=1)

    # 8.d) Save computational graph
    writer.add_graph(net, images)

    # 8.e) Save a sample of images
    img_grid = torchvision.utils.make_grid(images.reshape(-1, 1, 28, 28)[:64])
    writer.add_image('quickdraw_images', img_grid)

    writer.close()
    print("All done. Run 'tensorboard --logdir runs' to view.")

```


1.  **Task: `argparse`**: At the bottom of `train.py`, add the `if __name__ == '__main__':` block. Inside, you need to define an `argparse.ArgumentParser()` to accept command-line arguments. Define defaults for each argument.

      * **Required arguments**:
          * `--exp_name` 
          * `--epochs`
          * `--batch_size`
          * `--dropout_rate`
          * `--lr`
          * `--weight_decay`
      * After parsing, retrieve these args into variables (e.g., `epochs = args.epochs`).

    <details>
    <summary>🕵️‍♂️ Hint: `argparse`</summary>

    ```python
    import argparse
    parser = argparse.ArgumentParser(description='Train a ConvNet on Quick, Draw!')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    # ... add the rest ...
    args = parser.parse_args()
    ```

    </details>

2.  **Task: TensorBoard**: We need to see our training logs\!

      * Import `SummaryWriter`: `from torch.utils.tensorboard import SummaryWriter`
      * In your `main` block, initialize it: `writer = SummaryWriter(f'runs/{args.exp_name}')`
      * **Modify your `train` function** to log the scalars:
        ```python
        # ... end of epoch ...
        writer.add_scalar('Loss/train', mean(running_loss), epoch)
        writer.add_scalar('Loss/val', mean(val_loss), epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        ```
      * Modify your `tqdm` description to show both train and val loss.

3.  **Task: Save Model**: After the training loop finishes, save the weights.

    ```python
    import os
    # ... after train loop ...
    print(f'Test accuracy: {test(net, test_loader, device)}')
    if not os.path.exists('weights'):
        os.makedirs('weights')
    torch.save(net.state_dict(), f'weights/{args.exp_name}_net.pth')
    print('Model weights saved.')
    ```

## Step 9: Final Run and Visualization

You're ready.

1.  **Run the script** from your terminal (make sure your `.venv` is active\!):
    ```bash
    python train.py --exp_name ... --epochs ... 
    ```
2.  **Launch TensorBoard**:
    ```bash
    tensorboard --logdir runs
    ```
3.  Open `http://localhost:6006` in your browser.
4.  **Check the `Graph` tab**: You should see your `ConvNet` architecture.
5.  **Check the `Projector` tab**: Click the "inactive" button and select "projector". You should see your 10 classes starting to form distinct clusters. This is your model learning\!

## Step 10: Commit Your Neural Network

Now that you have a working, trained neural network, it's time to save your progress.

1. **Add all training-related files**:
   ```bash
   git add train.py model.py EDA.ipynb
   git commit -m "Complete neural network training pipeline with TensorBoard logging"
   ```

2. **Update `.gitignore`**: Add these lines to avoid committing training artifacts:
   ```
   runs/
   weights/
   .venv/
   ```

3. **Commit the gitignore update**:
   ```bash
   git add .gitignore
   git commit -m "Exclude training artifacts from version control"
   ```

-----

If you've made it this far, congratulations\! You've built, trained, and professionally logged a real deep-learning model from scratch.

In the next part, we'll take our saved `weights/{exp_name}_net.pth` file and serve it with a FastAPI API.