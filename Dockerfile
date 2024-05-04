ARG HF_TOKEN

FROM python:3.10-bullseye

RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install rust
RUN apt-get update && apt-get install -y curl

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Try and run pip command after setting the user with `USER user` to avoid permission issues with Python
RUN pip install --no-cache-dir --upgrade pip

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# expose the port
EXPOSE 7860

# set env variables
ENV HF_TOKEN=$HF_TOKEN

# Run the app
CMD ["python", "$HOME/app/app.py"]