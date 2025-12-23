# Part 1.6: Building a User Interface with Gradio

We have a trained model and an API to serve it. But unless you like using `curl` commands in a terminal, we need a Frontend.

We will use **Gradio**. It's a fantastic library for building Machine Learning demos in pure Python. It allows us to draw on a digital sketchpad and send that image to our API.

## Step 0: Create a New Branch

Let's create a new branch for frontend development:

```bash
git checkout main
git pull origin main
git checkout -b dev-frontend
```

## Step 1: Install Gradio

Update your `requirements.txt` to include `gradio` and `requests`:

```text
gradio==3.50.2
requests==2.32.4
```

*(Note: We fix versions to ensure compatibility).*

Run:

```bash
pip install -r requirements.txt
```

## Step 2: Create `frontend.py`

Create a file named `frontend.py`.

We have provided the User Interface code for you (the bottom part of the file). **Your mission is to write the logic function** that connects the UI to your API.

**Task**: Copy this code and fill in the `# TODO` blocks in the `predict_drawing` function.

```python
import gradio as gr
from PIL import Image
import requests
import io
import numpy as np

# --- LOGIC SECTION ---
def predict_drawing(image):
    """
    This function takes the image from the sketchpad, processes it,
    sends it to the API, and returns the prediction.
    """
    
    # 1. Handle Gradio input format (Gradio sometimes returns a dict)
    if isinstance(image, dict) and 'composite' in image:
        image = image['composite']
    
    # 2. Convert Numpy array to PIL Image
    if hasattr(image, 'astype'):
        image = Image.fromarray(image.astype('uint8'))
    
    # 3. Ensure grayscale
    try:
        if image.mode != 'L':
            image = image.convert('L')
    except Exception as e:
        print(f"Error converting image to grayscale: {e}")
        return "Error processing image"

    # 4. Invert colors
    # Sketchpad draws black on white (255 background, 0 ink).
    # Neural networks (like MNIST/QuickDraw) usually train on white ink on black background.
    image = Image.eval(image, lambda x: 255 - x)

    # 5. Save image to bytes to send via API
    img_binary = io.BytesIO()
    image.save(img_binary, format='PNG')
    img_binary = img_binary.getvalue()

    # TODO: Define your API URL
    # Since we are running locally, it should look like [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)
    api_url = "..." 

    # TODO: Send the POST request
    # Use requests.post()
    # Pass img_binary as data
    # Warning: The API expects a file upload, so you might need to check how to send raw bytes 
    # or wrap it in a 'files' dictionary depending on your API implementation.
    # For this tutorial, we assume the API handles raw bytes or multipart/form-data.
    try:
        # Hint: response = requests.post(url, files={"file": ...})
        response = ...
    except requests.exceptions.ConnectionError:
        return "Error: API is down. Is api.py running?"

    # TODO: Parse the response
    if response.ok:
        # Extract the JSON content
        # Return the 'class' or 'label' from the JSON
        return ...
    else:
        print(f"Request failed: {response.status_code} - {response.reason}")
        return "Prediction failed"


# --- UI SECTION (Don't touch this!) ---
if __name__=='__main__':
    # We define the interface
    interface = gr.Interface(
        fn=predict_drawing, 
        inputs="sketchpad", 
        outputs='label',
        live=False, # Set to True if you want real-time feedback (can be slow)
        title="Quick, Draw! Pictionary",
        description="Draw a cat, an airplane, or a donut. Click 'Submit' to see if the model recognizes it!",
    )
    
    # Launch the server
    # server_name='0.0.0.0' allows access from other machines if needed
    interface.launch(debug=True, share=False, server_name='0.0.0.0', server_port=7860)
```

### 🕵️‍♂️ Hints for the Logic

<details>
<summary>Sending files with `requests`</summary>

Your `api.py` uses `UploadFile`. The Python `requests` library handles this easily if you use the `files` parameter.

```python
# Example
response = requests.post(
    "http://127.0.0.1:8000/predict", 
    files={"file": ("drawing.png", img_binary, "image/png")}
)
```

</details>

<details>
<summary>Parsing the JSON</summary>

If your `api.py` returns: `{"class": "cat", "confidence": 0.95}`

You can extract it like this:

```python
data = response.json()
return data["class"]
# OR return the whole dictionary if Gradio output is set to 'label'
# return {data["class"]: data["confidence"]}
```

</details>

## Step 3: Launching the Stack

To test this, you need **two** terminals open.

1.  **Terminal 1 (The Brain):** Run the API.

    ```bash
    uvicorn api:app --reload
    ```

2.  **Terminal 2 (The Face):** Run the Frontend.

    ```bash
    python frontend.py
    ```

3.  Open the URL shown in Terminal 2 (usually `http://127.0.0.1:7860`) in your web browser.

4.  **Draw\!** Try drawing a banana. Does it work?

## Step 4: Commit and Create Pull Request

Your frontend is complete! Time to merge it.

1. **Commit your changes**:
   ```bash
   git add frontend.py requirements.txt
   git commit -m "Add Gradio frontend for model interaction"
   ```

2. **Push and create a Pull Request**:
   ```bash
   git push origin dev-frontend
   ```
   
3. **On GitHub**: Create a Pull Request from `dev-frontend` to `main`, review your changes, merge it, and then update your local `main` branch:
   ```bash
   git checkout main
   git pull origin main
   ```

-----

**Troubleshooting:**

  * If the prediction is always wrong, check the **color inversion**. If you drew a black line on white, and inverted it, you are sending a white line on black. Check if your `train.py` and `model.py` were trained on white-on-black images (standard for datasets like this). If your model was trained on inverted colors, you might not need `Image.eval`.

**Next Step:** Congratulations\! You have a full-stack AI application running on your machine. In the next part, we will **Dockerize** these two applications to make them portable.
