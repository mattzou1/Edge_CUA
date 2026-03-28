FROM dorowu/ubuntu-desktop-lxde-vnc:latest

ENV RESOLUTION=1280x800

# Install LibreOffice and gnome-terminal for demo tasks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    gnome-terminal \
    && rm -rf /var/lib/apt/lists/*
