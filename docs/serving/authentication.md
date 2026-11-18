# Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated.

A key passed with `--api-key` is visible to every local user in `ps`, `/proc/PID/cmdline` and `systemctl status`. To
keep it off the command line, use `--api-key-file FILE` (the file holds the key on one line; one trailing newline is
stripped; a missing or empty file stops the server at startup) or set `NINFER_API_KEY` (read only when neither flag
is given; with systemd, `EnvironmentFile=`). `--api-key` and `--api-key-file` together are refused.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.
