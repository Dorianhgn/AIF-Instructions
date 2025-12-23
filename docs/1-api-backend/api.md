# Part 1.5: Exposing the Model with FastAPI

Great job on Part 1! You have successfully trained a Convolutional Neural Network that can recognize drawings. You have the "brain" stored in a `.pth` file.

However, a model sitting in a file is not very useful to the world. Users won't run a Python script to check if they drew a cat; they want a web interface or an app.

In this part, we will wrap your PyTorch model inside a **FastAPI** application. This will allow anyone to send an image to your server and get a prediction back in JSON format.

## Step 1: Update Environment

We need new libraries to handle the web server and image processing.

1.  **Update `requirements.txt`**: Add the following lines to your existing file:

    ```text
    fastapi==0.103.1
    uvicorn==0.23.2
    python-multipart==0.0.6
    Pillow==10.0.0
    ```

2.  **Install them**:
    ```bash
    pip install -r requirements.txt
    ```

## Step 2: Understanding the API Structure

We are going to create a file named `api.py`. Here is the workflow we want to implement:

1.  **Startup**: When the server starts, it loads the neural network into memory **once**. We don't want to load the model for every single request (that would be too slow!).
2.  **Pre-processing**: The API will receive image files (JPEG, PNG). Our model expects a normalized PyTorch tensor of shape `(1, 1, 28, 28)`. We need a function to bridge this gap.
3.  **Prediction**: We run the tensor through the model.
4.  **Response**: We return the class name (e.g., "cat") and the confidence score.

## Step 3: Create `api.py`

Create a file named `api.py` at the root of your project.

**Task**: Copy the skeleton below and fill in the `# TODO` sections.

```python
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import torch
import io
from PIL import Image
from typing import List

# Import your model architecture
# Ensure model.py is in the same directory or properly in PYTHONPATH
from model import ConvNet

# Hardcoded classes (Must match the order of your training!)
CLASS_NAMES = [
    "cat", "dolphin", "moon", "alarm clock", "banana", 
    "airplane", "circle", "door", "donut", "eye"
]

# Global variables to store the model and device
model = None
device = None

# --- LIFESPAN (Startup & Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function executes when the API starts.
    Use it to load the model into memory.
    """
    global model, device
    
    # TODO: Set the device (cuda or cpu)
    # device = ...

    # TODO: Initialize the ConvNet model
    # model = ...
    
    # TODO: Load the trained weights
    # Hint: Use torch.load() and model.load_state_dict()
    # Make sure to map_location=device if you trained on GPU but run API on CPU
    model_path = "weights/your_exp_name_net.pth" # Update this path!
    print(f"Loading model from {model_path}...")
    
    # ... load state dict ...
    
    # TODO: Set model to eval mode
    # ...
    
    print("Model loaded successfully!")
    yield
    # Code here would run on shutdown (not needed for now)
    print("Shutting down...")

# --- API INITIALIZATION ---
app = FastAPI(title="Quick, Draw! API", lifespan=lifespan)

# --- PREPROCESSING ---
def transform_image(image_bytes):
    """
    Transforms raw image bytes into a PyTorch tensor.
    Steps:
    1. Open bytes with PIL
    2. Convert to Grayscale (L)
    3. Resize to 28x28
    4. Convert to Tensor and Normalize
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    # TODO: Convert to grayscale
    # image = ...
    
    # TODO: Resize to 28x28
    # image = ...
    
    # TODO: Convert to numpy/tensor, normalize to [0, 1], and add batch dimension
    # The final shape must be (1, 1, 28, 28)
    # tensor = ...
    
    return tensor.to(device)

# --- ENDPOINTS ---

@app.get("/")
def index():
    return {"message": "Welcome to the Quick, Draw! API. Use /predict to classify images."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives a single image file, processes it, and returns the prediction.
    """
    # TODO: Read the file content
    # contents = await file.read()
    
    # TODO: Transform the image using your helper function
    # input_tensor = ...
    
    # TODO: Make a prediction
    # with torch.no_grad():
    #     output = ...
    #     prediction_idx = ...
    #     confidence = ...
    
    # TODO: Return JSON response
    # return {
    #     "filename": file.filename,
    #     "class": ...,
    #     "confidence": ...
    # }
    return {"status": "not implemented yet"}

@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Receives a list of image files and returns a list of predictions.
    """
    results = []
    
    # TODO: Loop through the files
    # for file in files:
    #     ... (repeat logic or call a helper function) ...
    #     results.append(...)
    
    return {"results": results}
```

### 🕵️‍♂️ Hints for the code

<details>
<summary>How to transform the image?</summary>

The "Quick, Draw!" dataset is grayscale. However, uploaded images might be RGB.

1.  `image = image.convert('L')` converts to grayscale.

2.  `image = image.resize((28, 28))` handles the size.

3.  To make it a tensor: `torch.tensor(np.array(image))` or use `torchvision.transforms.ToTensor()`.

4.  **Crucial**: The model expects a batch of images `(Batch_Size, Channels, Height, Width)`. Even for a single image, you need to add the batch dimension. `tensor.unsqueeze(0)` is your friend.

</details>

<details>
<summary>How to read an UploadFile?</summary>

In FastAPI, `UploadFile` is asynchronous. You must use `await`:

```python
contents = await file.read()
```

Then pass `contents` to `io.BytesIO(contents)` to make it look like a real file object for Pillow.

</details>

<details>
<summary>How to get the class name from prediction?</summary>

The model returns raw logits (scores).

1.  `probs = torch.softmax(output, dim=1)` gives you probabilities.
2.  `pred_index = torch.argmax(probs, dim=1).item()` gives the index of the highest score.
3.  `CLASS_NAMES[pred_index]` gives the string label.

</details>

## Step 4: Run the Server

Once you have filled in the blanks, it's time to launch your API. We use `uvicorn`, which is an ASGI server (a fast web server for Python).

1.  Open your terminal.

2.  Make sure your virtual environment is active.

3.  Run:

    ```bash
    uvicorn api:app --reload
    ```

      * `api`: The name of your python file (`api.py`).
      * `app`: The name of the FastAPI instance inside that file.
      * `--reload`: Makes the server restart automatically when you save code changes (useful for dev).

4.  You should see something like: `Uvicorn running on http://127.0.0.1:8000`.

## Step 5: Test with Swagger UI

FastAPI provides automatic documentation.

1.  Open your browser at [http://127.0.0.1:8000/docs](https://www.google.com/search?q=http://127.0.0.1:8000/docs).
2.  You will see the "Swagger UI".
3.  Click on **POST /predict**.
4.  Click **Try it out**.
5.  Upload an image (try to find a simple drawing of a cat or an apple on Google, or draw one in Paint and save as PNG).
6.  Click **Execute**.
7.  Check the "Server response". Did it guess correctly?

## Step 6: Commit Your API

Now that your API is working, commit your changes:

```bash
git add api.py requirements.txt
git commit -m "Implement FastAPI server for model inference"
```

## Step 7: Create a Pull Request

Excellent work! You've completed the backend development. Now it's time to merge your work into the `main` branch.

### Why Pull Requests?

Pull Requests (PRs) are a professional way to:
- Review code before merging
- Document what changed and why
- Keep a clean project history
- Catch bugs before they reach production

### Create Your Pull Request

1. **Push your branch to GitHub**:
   ```bash
   git push origin dev-backend
   ```

2. **Open GitHub in your browser**:
   - Go to your repository on GitHub
   - You should see a yellow banner saying "dev-backend had recent pushes" with a **Compare & pull request** button
   - Click that button

3. **Fill in the Pull Request form**:
   - **Title**: "Add neural network training and FastAPI server"
   - **Description**: Write a brief summary of what you built:
     ```
     ## Changes
     - Implemented ConvNet architecture for Quick, Draw! dataset
     - Created training pipeline with TensorBoard logging
     - Built FastAPI server to serve model predictions
     - Added proper requirements management
     
     ## Testing
     - Model achieves ~XX% accuracy on test set
     - API tested with Swagger UI
     ```

4. **Create the Pull Request**: Click the green "Create pull request" button

5. **Merge the Pull Request**:
   - Review your changes one last time
   - Click "Merge pull request"
   - Click "Confirm merge"
   - Optionally, delete the `dev-backend` branch on GitHub

6. **Update your local repository**:
   ```bash
   git checkout main
   git pull origin main
   ```

Congratulations! Your backend is now officially part of the main codebase.

-----

### Challenge (Optional)

In `batch_predict`, processing images one by one inside a `for` loop is okay, but it's not efficient for the GPU. Can you rewrite `batch_predict` to stack all image tensors into a single batch tensor (e.g., shape `(N, 1, 28, 28)`) and run `model(batch_tensor)` only once?

-----

**Next Step:** Now that our API works locally, we need to create a user-friendly interface. In the next part, we will build a simple web UI using **Gradio** that interacts with this API. After that, we will Dockerize both the API and the UI for easy deployment!
