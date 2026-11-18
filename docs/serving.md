# HTTP serving

`build/apps/ninfer-serve` loads one v3 `.ninfer` artifact and exposes OpenAI- and
Anthropic-compatible HTTP endpoints over one resident NInfer Engine. For capy-flash's models on 16 GB boards
start with [capy-flash serving](serving/capy-flash.md); the sections after it are the general server reference.

## Contents

The server reference is split into one page per topic under [`serving/`](serving/). Every section title that
was an anchor of this page still is one: it leads to the section's new place.

- <a id="capy-flash-serving"></a>[capy-flash serving](serving/capy-flash.md)
  - <a id="launchers-and-profiles"></a>[Launchers and profiles](serving/capy-flash.md#launchers-and-profiles)
    - <a id="profiles"></a>[Profiles](serving/capy-flash.md#profiles)
    - <a id="vision"></a>[Vision](serving/capy-flash.md#vision)
  - <a id="ram"></a>[RAM](serving/capy-flash.md#ram)
  - <a id="chat-template"></a>[Chat template](serving/capy-flash.md#chat-template)
  - <a id="cpu-expert-threads"></a>[CPU expert threads](serving/capy-flash.md#cpu-expert-threads)
  - <a id="connect-a-client"></a>[Connect a client](serving/capy-flash.md#connect-a-client)
  - <a id="what-the-engine-does"></a>[What the engine does](serving/capy-flash.md#what-the-engine-does)
- <a id="start-the-server"></a>[Start the server](serving/start.md)
- <a id="endpoints"></a>[Endpoints](serving/endpoints.md)
- <a id="openai-chat-completions"></a>[OpenAI Chat Completions](serving/chat-completions.md)
  - <a id="llamacpp-compatible-request-observations"></a>[llama.cpp-compatible request observations](serving/chat-completions.md#llamacpp-compatible-request-observations)
  - <a id="multimodal-request"></a>[Multimodal request](serving/chat-completions.md#multimodal-request)
- <a id="openai-prompt-caching"></a>[OpenAI prompt caching](serving/prompt-caching.md)
- <a id="openai-responses-core"></a>[OpenAI Responses Core](serving/responses.md)
  - <a id="create-a-response"></a>[Create a Response](serving/responses.md#create-a-response)
  - <a id="create-request-fields"></a>[Create request fields](serving/responses.md#create-request-fields)
  - <a id="input-item-contract"></a>[Input Item contract](serving/responses.md#input-item-contract)
  - <a id="function-tools"></a>[Function tools](serving/responses.md#function-tools)
  - <a id="response-object-and-usage"></a>[Response object and usage](serving/responses.md#response-object-and-usage)
  - <a id="responses-streaming"></a>[Responses streaming](serving/responses.md#responses-streaming)
  - <a id="local-response-state-and-resources"></a>[Local response state and resources](serving/responses.md#local-response-state-and-resources)
  - <a id="responses-input-token-count"></a>[Responses input token count](serving/responses.md#responses-input-token-count)
- <a id="anthropic-messages"></a>[Anthropic Messages](serving/anthropic-messages.md)
- <a id="structured-output"></a>[Structured output](serving/structured-output.md)
  - <a id="request-fields"></a>[Request fields](serving/structured-output.md#request-fields)
  - <a id="errors"></a>[Errors](serving/structured-output.md#errors)
  - <a id="structured-output-limits"></a>[Structured output limits](serving/structured-output.md#structured-output-limits)
  - <a id="cost"></a>[Cost](serving/structured-output.md#cost)
- <a id="authentication-and-cors"></a>[Authentication and CORS](serving/authentication.md)
- <a id="server-options"></a>[Server options](serving/options.md)
  - <a id="disk-tier-of-the-context-cache"></a>[Disk tier of the context cache](serving/options.md#disk-tier-of-the-context-cache)
  - <a id="two-gpus"></a>[Two GPUs](serving/options.md#two-gpus)
- <a id="structured-request-log"></a>[Structured request log](serving/request-log.md)
- <a id="execution-behavior"></a>[Execution behavior](serving/execution.md)
