# nvrx-nodestate-tunnel-proxy

`nvrx-nodestate-tunnel-proxy` is an ingress bridge for environments where
`nvrx-nodestatesvc` must run somewhere without inbound connectivity.

The proxy runs nginx and sshd:

- nginx serves `/edge-healthz` directly.
- nginx forwards all other HTTP requests to `127.0.0.1:18080`.
- sshd accepts a dedicated public-key-only `tunnel` user.
- the `tunnel` user may open remote TCP forwards, but cannot get an interactive
  shell.

A host with Slurm access starts `nvrx-nodestatesvc` locally and opens a reverse
SSH tunnel:

```text
ssh -N -R 127.0.0.1:18080:127.0.0.1:18080 tunnel@PROXY -p 2222
```

After the tunnel is established, HTTP requests to the proxy are forwarded to
the host running `nvrx-nodestatesvc`.

Deployment-specific scripts, Kubernetes manifests, image names, and secrets are
not kept here; they belong to the operator workspace for the target cluster.
