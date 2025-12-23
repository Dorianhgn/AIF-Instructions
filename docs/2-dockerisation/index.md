# Part 2: Dockerization (Shipping our App)

We have an API and a Frontend working on your machine. But if you send your code to a friend, they might have a different version of Python, different drivers, or a different OS. The infamous *"It works on my machine"* problem.

To solve this, we use **Docker**.

## Step -1: Create a New Branch

Start by creating a new branch for dockerization:

```bash
git checkout main
git pull origin main
git checkout -b dev-docker
```

## Step 0: What is Docker?

Before we code, watch this short video. It explains the difference between a **Dockerfile** (the recipe), an **Image** (the cake mold), and a **Container** (the actual cake).


<iframe width="560" height="315" src="https://www.youtube.com/embed/Gjnup-PuquQ?si=3s9lueubJv1bnhJE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


* **The key takeaways:**
    * **Dockerfile**: A text file with instructions.
    * **Image**: The built artifact (read-only).
    * **Container**: A running instance of an image.

## Step 1: Prepare the requirements

Right now, you have one big `requirements.txt`. To keep our Docker images light, we should split them.

1.  **Task**: Create a file `requirements-api.txt` with only what the API needs:
    ```text
    fastapi==0.103.1
    uvicorn==0.23.2
    python-multipart==0.0.6
    Pillow==10.0.0
    torch==2.0.1
    torchvision==0.15.2 
    ```

2.  **Task**: Create a file `requirements-frontend.txt` with only what Gradio needs:
    ```text
    gradio==3.50.2
    requests==2.32.4
    Pillow==10.0.0
    ```

## Step 2: The API Dockerfile (The "Naive" Approach)

We are going to write our first recipe to package the API.

1.  Create a file named `Dockerfile` (no extension) in your project root.
2.  **Task**: Copy the following content into it.

```dockerfile
# 1. Use an official Python runtime as the parent image
FROM python:3.10-slim

# 2. Set the working directory in the container to /app
WORKDIR /app

# 3. Copy everything from your current directory to /app in the container
COPY . /app
# `COPY . .` works too, but is less explicit

# 4. Install any needed packages
RUN pip install --no-cache-dir -r requirements-api.txt

# 5. Make port 5075 available to the world outside this container
EXPOSE 5075

# 6. Run the API when the container launches
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5075"]
```

### Experiment: Why is this "Bad"?

Let's verify how Docker builds this image.

1.  **Build the image**: Run this command in your terminal.

    ```bash
    docker build -t my-api .
    ```

    *Watch it run. It installs torch, torchvision... it takes some time.*

2.  **Make a change**: Later on, when we will use `docker-compose`, we need to make sure our Python code uses that `API_URL` environment variable. We handled the case where the env variable is not set, so our code works both in a Docker container, and in docker-compose.

    1.  Open `frontend.py`.
    2.  **Task**: Modify the URL definition in your `predict_drawing` function:

    <!-- end list -->

    ```python
    import os

    # ... inside predict_drawing function ...

    # If running in Docker, we use the env variable http://api:5075
    # If running locally, we default to [http://127.0.0.1:5075](http://127.0.0.1:5075)
    api_url = os.getenv("API_URL", "[http://127.0.0.1:5075](http://127.0.0.1:5075)")
    url = f"{api_url}/predict"

    # ... proceed with requests.post ...
    ```

3.  **Build again**: Run the same command:

    ```bash
    docker build -t my-api .
    ```

**Observation**: Did you notice? **It is running `pip install` again\!** 

Why? Because Docker builds in **Layers**.

  * In the file above, we said `COPY . /app`.
  * Since you changed `api.py`, Docker sees that the inputs for the `COPY` layer have changed.
  * Consequently, it invalidates that layer **and every layer after it**.
  * So, it must re-run `RUN pip install ...`.

In MLOps, you change your code 100 times a day. You don't want to wait 2 minutes for dependencies to reinstall every time you fix a typo\!

## Step 2.5: The API Dockerfile (The "Optimized" Approach)

Let's fix this by being smarter about the order of operations. **We want to copy the requirements *first*, install them, and *then* copy the code.**

1.  **Task**: Update your `Dockerfile` with the optimized version where we copy the requirements first.

    <details>

    <summary>👀 Solution</summary>

    <!-- end list -->

    ```dockerfile
    # Use an official Python runtime
    FROM python:3.10-slim

    # Set work directory
    WORKDIR /app

    # --- THE MAGIC PART ---
    # 1. Copy ONLY the requirements first
    COPY requirements-api.txt .

    # 2. Install dependencies (including gdown for model weights)
    # Since requirements-api.txt hasn't changed, Docker will use the CACHE for this layer!
    RUN pip install --no-cache-dir -r requirements-api.txt && \
        pip install --no-cache-dir gdown

    # 3. Download model weights
    # Replace YOUR_GOOGLE_DRIVE_FILE_ID with your actual file ID
    RUN mkdir -p weights && \
        gdown --id YOUR_GOOGLE_DRIVE_FILE_ID -O weights/your_exp_name_net.pth

    # 4. Copy the rest of the code
    # This layer will be rebuilt, but it's super fast (just copying files)
    COPY . .
    # ----------------------

    EXPOSE 5075

    CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5075"]
    ```

    </details>

2.  **Important: Upload your model weights**:
      * Train your model and save the weights locally
      * Upload `weights/your_exp_name_net.pth` to Google Drive
      * Get the sharing link and extract the file ID (the part after `/d/` and before `/view`)
      * Replace `YOUR_GOOGLE_DRIVE_FILE_ID` in the Dockerfile with your actual ID
      * Update `your_exp_name_net.pth` to match your actual filename

3.  **Verify**:
      * Build (`docker build -t my-api .`). (First time might be slow as structure changed).
      * Change a comment in `api.py`.
      * Build again. **Boom\! Instantaneous.** 

## Step 3: The Frontend Dockerfile

Now it's your turn.

1.  **Task**: Create a file named `Dockerfile-frontend`.
2.  **Task**: Write the Dockerfile to serve the frontend.

**Constraints:**

  * Use `requirements-frontend.txt`.
  * Use the **Optimized** approach (install requirements before copying code).
  * Expose port `7860` (Gradio's default).
  * The command is `python frontend.py`.

<details>
<summary>👀 Solution</summary>

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements-frontend.txt .
RUN pip install --no-cache-dir -r requirements-frontend.txt
COPY . .
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "frontend.py"]
```

</details>

3. **Commit your Dockerfiles**:
   ```bash
   git add Dockerfile Dockerfile-frontend requirements-api.txt requirements-frontend.txt
   git commit -m "Add Dockerfiles for API and frontend services"
   ```

## Step 4: Orchestrating with Docker Compose

### 🎻 Analogy: The Conductor

Think of your application as an **Orchestra**.

* **The Dockerfile** is the **Sheet Music** for a specific instrument (e.g., "How to be a Violin").
* **The Container** is the **Musician** playing that instrument.
* **Docker Compose** is the **Conductor**.

The Conductor doesn't play the instruments. Instead, they look at a plan (the `docker-compose.yml` file) and say: *"You (API) sit here, you (Frontend) sit there, both of you start playing at the same time, and Frontend, listen to the API's tempo."*

### 1. The `docker-compose.yml` file

Create a file named `docker-compose.yml` at the root of your project.

**Task**: Copy this content. Read the comments to understand what the Conductor is doing.

```yaml
version: '3.8'

services:
  # --- Musician 1: The Brain (API) ---
  api: 
    build: 
      context: .
      dockerfile: Dockerfile
    ports:
      - "5075:5075"  # Opens a window so we can hear it (Host Port : Container Port)

  # --- Musician 2: The Face (Frontend) ---
  gradio-app:
    build: 
      context: .
      dockerfile: Dockerfile-frontend
    ports:
      - "7860:7860" # Opens the port for the web interface
    depends_on:
      - api  # Tells Docker: "Don't start the Frontend until the API is running"
    environment:
      # This is the magic link!
      # Inside this network, the computer name of the api is simply "api".
      - API_URL=http://api:5075
```

### 2\. Docker Compose Cheat Sheet

Here are the only 4 commands you need to know to be a Conductor.

#### Start the Concert

```bash
docker-compose up
```

  * **What it does:** Builds the images (if they don't exist), creates the network, and starts all containers.
  * **Use case:** When you want to see the logs of all services streaming in your terminal.

#### Start in the Background ("Detached")

```bash
docker-compose up -d
```

  * **What it does:** Starts everything, but gives you your terminal back immediately. The containers run in the background.
  * **Use case:** When you are done debugging and just want the app running while you do other things.

#### 🏗️ Force Rebuild (Important\!)

```bash
docker-compose up --build
```

  * **What it does:** Forces Docker to read your Dockerfiles and rebuild the images before starting.
  * **Use case:** **Use this whenever you change your Python code\!** If you edit `api.py` and just run `docker-compose up`, Docker will use the old version of the code stored in the container. You must `--build` to update it.

#### Stop Everything

```bash
docker-compose down
```

  * **What it does:** Stops the containers and removes the network.
  * **Use case:** When you are finished working.



## Step 5: Launch and Test

1.  `docker-compose up --build`
2.  Go to `http://localhost:7860`.
3.  Enjoy your fully containerized MLOps project\!

4. **Commit your orchestration**:
   ```bash
   git add docker-compose.yml
   git commit -m "Add Docker Compose orchestration for full-stack deployment"
   ```

5. **Update the README**: Now that your project is fully containerized, update your `README.md` with deployment instructions. 
  Replace the content with something like **[this](https://github.com/eddaiveland/movie-screening-platform/blob/c5070119b469a6a4bbb6178da24a125b4d0fda79/README.md)**.

6. **Commit the README**:
   ```bash
   git add README.md
   git commit -m "Update README with complete deployment instructions"
   ```

## Step 6: Create Pull Request

Your application is now fully containerized! Let's merge this work.

1. **Push and create a Pull Request**:
   ```bash
   git push origin dev-docker
   ```

2. **On GitHub**: Create a Pull Request from `dev-docker` to `main`, merge it, and update your local `main`:
   ```bash
   git checkout main
   git pull origin main
   ```

Congratulations! You now have a fully version-controlled, containerized machine learning application ready for deployment.

<!-- end list -->

