FROM debian:bookworm-slim

# Keep the tunnel account unlocked for OpenSSH public-key auth; sshd_config
# disables password and keyboard-interactive auth.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        nginx \
        openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/sh tunnel \
    && passwd -d tunnel \
    && mkdir -p /run/sshd /home/tunnel/.ssh \
    && chown -R tunnel:tunnel /home/tunnel/.ssh \
    && chmod 700 /home/tunnel/.ssh

COPY nginx.conf /etc/nginx/nginx.conf
COPY sshd_config /etc/ssh/sshd_config
COPY entrypoint.sh /usr/local/bin/nvrx-nodestate-tunnel-proxy

RUN chmod +x /usr/local/bin/nvrx-nodestate-tunnel-proxy

EXPOSE 8000 2222
ENTRYPOINT ["/usr/local/bin/nvrx-nodestate-tunnel-proxy"]
