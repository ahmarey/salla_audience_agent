# Stage 1: Build stage - Install dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Create a virtual environment
RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy the requirements file
COPY requirements.txt .

# Install dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: Final stage - Create the production image
FROM python:3.12-slim

WORKDIR /app

ARG GOOGLE_API_KEY
ENV GOOGLE_API_KEY=$GOOGLE_API_KEY

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv

# Set path to include the virtual environment's binaries
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application code
COPY ./app ./app
RUN echo "--- Listing files in /app at BUILD time ---" && ls -la

# Expose the port the app runs on
EXPOSE 8000

# Run the application with Uvicorn for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
# CMD ["sh", "-c", "echo '--- Environment variables at RUN time ---' && printenv && echo '--- Container is now sleeping ---' && sleep 3600"]
