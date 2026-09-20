# Setting up a Mac to `ssh pod`

A Mac needs two things: an SSH key RunPod knows about, and a `Host pod` block
in `~/.ssh/config`. Nothing else lives on the Mac — no repo, no Python, no
credentials. Ten minutes, once per Mac.

## 1. Make a key on the new Mac

```bash
ssh-keygen -t ed25519 -C "$(scutil --get ComputerName)"   # accept the default path
cat ~/.ssh/id_ed25519.pub                                  # copy this line
```

**One key per Mac. Do not copy the private key from the other Mac.** A key
that exists on one machine can be revoked when that machine is lost; a key
that was AirDropped around cannot.

## 2. Register the public key with RunPod

Console → **Settings → SSH Public Keys**. Paste the new line *below* the
existing one — one key per line — and save.

RunPod injects every registered key into a pod **when the pod is created**.
So the new Mac works on every pod from now on, but not on one that is already
running. To get into a running pod, append the key there by hand, from a Mac
that already has access:

```bash
ssh pod 'cat >> ~/.ssh/authorized_keys' < new-mac.pub
```

(or paste the public key to Claude on the pod and ask it to append it — a
public key is safe to paste anywhere).

## 3. Add the `pod` host

On the pod, print a ready-made block for the pod you are on:

```bash
make ssh-config
```

Paste the output into `~/.ssh/config` on the Mac. It looks like this:

```
Host pod
  HostName 203.0.113.7          # changes with every pod
  Port 10341                    # changes with every pod
  User root
  IdentityFile ~/.ssh/id_ed25519
  # RunPod's TCP proxy reaps idle connections, and a session is idle while
  # Claude Code thinks. Keepalives from this end stop the broken pipes.
  ServerAliveInterval 30
  ServerAliveCountMax 6
  TCPKeepAlive yes
  # pods are disposable and their IPs recycle: a remembered host key is a
  # guaranteed false alarm on the next pod
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
```

## 4. Connect

```bash
ssh pod
tmux new -A -s work        # or: herdr   (not both — they share Ctrl-b)
```

## Every new pod: update two lines

A pod is terminated when you stand up, and the next one gets a new IP and
port. On **each** Mac, change `HostName` and `Port` — the values are in the
console under **Connect → SSH over exposed TCP**, or from `make ssh-config`
once you are in from any machine. Everything else in the block stays.

## Two Macs, one pod

Both can be connected at once. `tmux new -A -s work` from the second Mac
attaches to the same session, so you see the same screen from both. What to
avoid is two *writers*: one Claude session editing the repo from each Mac will
collide exactly like two people would.

## If it does not connect

| Symptom | Cause |
|---|---|
| `Permission denied (publickey)` | the key was registered after this pod was created — step 2, second half |
| `Connection refused` / timeout | stale `HostName` / `Port` from a previous pod |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | the two host-key lines from step 3 are missing |
| `Broken pipe` after a few idle minutes | the three keepalive lines are missing |
