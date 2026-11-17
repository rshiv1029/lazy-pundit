import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "vitest";
import {
  baseUrlOf,
  type Config,
  doctor,
  draft,
  draftData,
  HUGGING_FACE_API,
  LlmError,
  modelOf,
  OLLAMA_API,
  openAiCompatibleLlm,
  readConfig,
  routedLlm,
  saveCredentials,
  type Transport,
  type TransportRequest,
} from "../src/index.ts";
import { scriptedLlm } from "./support/llm.ts";
import {
  committed,
  fearTheTurtle,
  noClipboard,
  operator,
} from "./support/workspace.ts";

/** A transport that answers every request with one status and body: JSON unless a string. */
function answering(status: number, body: unknown) {
  const requests: TransportRequest[] = [];
  return {
    requests,
    transport: {
      async request(req: TransportRequest) {
        requests.push(req);
        return {
          status,
          body: new TextEncoder().encode(
            typeof body === "string" ? body : JSON.stringify(body),
          ),
        };
      },
    },
  };
}

/** A streamed reply: each event as a `data:` line, then the end marker. */
function sse(...events: unknown[]): string {
  return `${[...events, "[DONE]"]
    .map((e) => `data: ${typeof e === "string" ? e : JSON.stringify(e)}\n\n`)
    .join("")}`;
}

const piece = (content: string, finish_reason: string | null = null) => ({
  choices: [{ delta: { content }, finish_reason }],
});

const usageChunk = {
  choices: [],
  usage: { prompt_tokens: 812, completion_tokens: 95 },
};

const ASK = {
  key: "hf_test",
  provider: "huggingface" as const,
  baseUrl: HUGGING_FACE_API,
  model: "openai/gpt-oss-120b",
  system: "You are a pundit.",
  prompt: "Week 8.",
  maxTokens: 8000,
};

const LOCAL = {
  ...ASK,
  key: "",
  provider: "openai-compatible" as const,
  baseUrl: OLLAMA_API,
  model: "qwen2.5:7b",
};

test("asks for a streamed reply with the system prompt and one user message", async () => {
  const { requests, transport } = answering(
    200,
    sse(
      piece("<think>hmm</think>## {intro}\n"),
      piece("Big week."),
      piece("", "stop"),
      usageChunk,
    ),
  );

  const reply = await openAiCompatibleLlm(transport).write(ASK);

  expect(reply).toEqual({
    text: "## {intro}\nBig week.",
    usage: { inputTokens: 812, outputTokens: 95 },
  });
  const [req] = requests as [TransportRequest];
  expect(req.url).toBe(`${HUGGING_FACE_API}/chat/completions`);
  expect(req.method).toBe("POST");
  expect(req.headers).toMatchObject({ authorization: "Bearer hf_test" });
  expect(JSON.parse(req.body ?? "")).toEqual({
    model: "openai/gpt-oss-120b",
    max_tokens: 8000,
    stream: true,
    stream_options: { include_usage: true },
    messages: [
      { role: "system", content: "You are a pundit." },
      { role: "user", content: "Week 8." },
    ],
  });
});

test("a server with no key gets no authorization header, and a trailing slash on the base is fine", async () => {
  const { requests, transport } = answering(
    200,
    sse(piece("Hi.", "stop"), usageChunk),
  );

  await openAiCompatibleLlm(transport).write({
    ...LOCAL,
    baseUrl: "http://localhost:11434/v1/",
  });

  const [req] = requests as [TransportRequest];
  expect(req.url).toBe("http://localhost:11434/v1/chat/completions");
  expect(req.headers).not.toHaveProperty("authorization");
});

test("a server that ignores stream and sends a whole reply still works", async () => {
  const { transport } = answering(200, {
    choices: [{ message: { content: "Whole." }, finish_reason: "stop" }],
    usage: { prompt_tokens: 3, completion_tokens: 4 },
  });

  expect(await openAiCompatibleLlm(transport).write(LOCAL)).toEqual({
    text: "Whole.",
    usage: { inputTokens: 3, outputTokens: 4 },
  });
});

async function errorFor(
  status: number,
  body: unknown,
  ask: typeof ASK | typeof LOCAL = ASK,
): Promise<LlmError> {
  const { transport } = answering(status, body);
  const err = await openAiCompatibleLlm(transport)
    .write(ask)
    .catch((e: unknown) => e);
  expect(err).toBeInstanceOf(LlmError);
  return err as LlmError;
}

test.each([
  [401, "invalid-key"],
  [403, "invalid-key"],
  [402, "no-credits"],
  [429, "provider"],
  [503, "provider"],
] as const)(
  "a %i answer is %s, named for Hugging Face",
  async (status, kind) => {
    const err = await errorFor(status, { error: "nope" });
    expect(err.kind).toBe(kind);
    expect(err.message).toContain(`Hugging Face answered ${status}: nope`);
  },
);

test("a local server's 404 says to pull the model", async () => {
  const err = await errorFor(
    404,
    { error: { message: "model 'qwen2.5:7b' not found" } },
    LOCAL,
  );
  expect(err.message).toContain(
    "localhost:11434 answered 404: model 'qwen2.5:7b' not found",
  );
  expect(err.message).toContain("ollama pull qwen2.5:7b");
});

test("a hosted server's 404 gets no Ollama advice", async () => {
  const err = await errorFor(
    404,
    { error: "no such model" },
    {
      ...ASK,
      baseUrl: "https://openrouter.ai/api/v1",
    },
  );
  expect(err.message).toContain("openrouter.ai answered 404");
  expect(err.message).not.toContain("ollama");
});

test("a server that isn't running is a provider error that says to start it", async () => {
  const transport: Transport = {
    async request() {
      throw new Error("fetch failed");
    },
  };
  const err = await openAiCompatibleLlm(transport)
    .write(LOCAL)
    .catch((e: unknown) => e);
  expect(err).toBeInstanceOf(LlmError);
  expect((err as LlmError).kind).toBe("provider");
  expect((err as LlmError).message).toMatch(
    /Couldn't reach localhost:11434: fetch failed.*ollama pull qwen2.5:7b/,
  );
});

test("a reply that stops short throws, with the tokens it used", async () => {
  for (const body of [
    sse(piece("Half a"), piece("", "length"), usageChunk),
    // The connection closed before the server said it was done.
    sse(piece("Half a"), usageChunk),
  ]) {
    const err = await errorFor(200, body, LOCAL);
    expect(err.message).toMatch(/stopped early/);
    expect(err.usage).toEqual({ inputTokens: 812, outputTokens: 95 });
  }
});

test("an error sent mid-stream throws its message", async () => {
  const err = await errorFor(
    200,
    sse(piece("Half a"), { error: { message: "model crashed" } }),
    LOCAL,
  );
  expect(err.message).toContain("reported an error: model crashed");
});

test("a body that isn't a chat completion throws naming the URL", async () => {
  const err = await errorFor(200, "<html>proxy page</html>");
  expect(err.message).toContain(`${HUGGING_FACE_API}/chat/completions`);
});

test("routedLlm sends Anthropic's requests to Anthropic and the others here", async () => {
  const { requests, transport } = answering(
    200,
    sse(piece("Hi.", "stop"), usageChunk),
  );
  await routedLlm(transport).write(LOCAL);
  expect((requests[0] as TransportRequest).url).toBe(
    "http://localhost:11434/v1/chat/completions",
  );
  const anthropic = answering(401, { error: { message: "bad key" } });
  await routedLlm(anthropic.transport)
    .write({ ...LOCAL, provider: "anthropic" })
    .catch(() => undefined);
  expect((anthropic.requests[0] as TransportRequest).url).toContain(
    "api.anthropic.com",
  );
});

async function week8Reply(): Promise<string> {
  const data = draftData(await committed(8), 8);
  return [
    `## {intro}\nWhat a week. ${data.intro}`,
    ...Object.entries(data.teams).map(
      ([id, t]) => `## {team=${id}}\nSteady as ever. ${t.statLine}`,
    ),
  ].join("\n\n");
}

async function withProvider(yaml: string) {
  const ws = await fearTheTurtle(8);
  const llm = scriptedLlm(await week8Reply());
  ws.deps.llm = llm;
  await mkdir(ws.deps.configDir, { recursive: true });
  await writeFile(join(ws.deps.configDir, "config.yaml"), `format: 1\n${yaml}`);
  return { ...ws, llm };
}

test("each provider's default model and base, and a named model wins", async () => {
  const { deps } = await withProvider("provider: huggingface\n");
  const hf = (await readConfig(deps)) as Config;
  expect([modelOf(hf), baseUrlOf(hf)]).toEqual([
    "openai/gpt-oss-120b",
    HUGGING_FACE_API,
  ]);
  const local = { ...hf, provider: "openai-compatible" as const };
  expect([modelOf(local), baseUrlOf(local)]).toEqual([
    "qwen2.5:7b",
    OLLAMA_API,
  ]);
  expect(
    baseUrlOf({ ...local, base_url: "https://api.groq.com/openai/v1" }),
  ).toBe("https://api.groq.com/openai/v1");
  expect(modelOf({ ...local, model: "llama3.1:8b" })).toBe("llama3.1:8b");
  expect(baseUrlOf({ ...hf, provider: "anthropic" })).toBeUndefined();
});

test("a base_url that isn't http or https is refused", async () => {
  const { deps } = await withProvider(
    "provider: openai-compatible\nbase_url: file:///etc/passwd\n",
  );
  await expect(readConfig(deps)).rejects.toThrow(/base_url/);
});

test("draft on provider openai-compatible needs no key, and sends the local base and model", async () => {
  const { dir, deps, llm } = await withProvider(
    "provider: openai-compatible\n",
  );

  const result = await draft(dir, deps);

  expect(llm.requests).toHaveLength(1);
  expect(llm.requests[0]).toMatchObject({
    key: "",
    provider: "openai-compatible",
    baseUrl: OLLAMA_API,
    model: "qwen2.5:7b",
    maxTokens: 8000,
  });
  expect(await readFile(result.path, "utf8")).toMatch(/Steady as ever\./);
});

test("draft sends LLM_API_KEY to an OpenAI-compatible server that wants one", async () => {
  const { dir, deps, llm } = await withProvider(
    "provider: openai-compatible\nbase_url: https://api.groq.com/openai/v1\nmodel: llama-3.3-70b-versatile\n",
  );
  deps.env.llmApiKey = "gsk_env";

  await draft(dir, deps);

  expect(llm.requests[0]).toMatchObject({
    key: "gsk_env",
    baseUrl: "https://api.groq.com/openai/v1",
    model: "llama-3.3-70b-versatile",
  });
});

test("draft on provider huggingface sends the HF_TOKEN", async () => {
  const { dir, deps, llm } = await withProvider("provider: huggingface\n");
  deps.env.hfToken = "hf_env";

  await draft(dir, deps);

  expect(llm.requests[0]).toMatchObject({
    key: "hf_env",
    provider: "huggingface",
    baseUrl: HUGGING_FACE_API,
    model: "openai/gpt-oss-120b",
  });
});

test("draft reads hf_token and llm_api_key from credentials.yaml", async () => {
  const hf = await withProvider("provider: huggingface\n");
  await saveCredentials(hf.deps, { huggingFaceToken: "hf_stored" });
  await draft(hf.dir, hf.deps);
  expect(hf.llm.requests[0]).toMatchObject({ key: "hf_stored" });

  const other = await withProvider("provider: openai-compatible\n");
  await saveCredentials(other.deps, { llmApiKey: "sk_stored" });
  await draft(other.dir, other.deps);
  expect(other.llm.requests[0]).toMatchObject({ key: "sk_stored" });
});

test("a call on another provider isn't counted against the spend cap", async () => {
  const { dir, deps, llm } = await withProvider(
    "provider: openai-compatible\nspend_cap_usd: 0\n",
  );

  await draft(dir, deps);

  expect(llm.requests).toHaveLength(1);
});

test("provider huggingface with no token falls back to stat lines and says how to set one", async () => {
  const { dir, deps, llm } = await withProvider("provider: huggingface\n");

  const result = await draft(dir, deps);

  expect(llm.requests).toHaveLength(0);
  expect(result.fallback).toMatch(/No Hugging Face token.*HF_TOKEN.*hf_token/);
});

test("a Hugging Face token with the default provider says to set provider: huggingface", async () => {
  const ws = await fearTheTurtle(8);
  ws.deps.llm = scriptedLlm("unused");
  ws.deps.env.hfToken = "hf_env";

  const result = await draft(ws.dir, ws.deps);

  expect(result.fallback).toMatch(/No Anthropic key.*provider: huggingface/);
});

test("with no key at all, a config that can't be read still gives stat lines", async () => {
  const ws = await fearTheTurtle(8);
  ws.deps.llm = scriptedLlm("unused");
  await mkdir(ws.deps.configDir, { recursive: true });
  await writeFile(
    join(ws.deps.configDir, "config.yaml"),
    "format: 1\nprovider: nonsense\n",
  );

  const result = await draft(ws.dir, ws.deps);

  expect(result.fallback).toMatch(/No Anthropic key/);
});

/** One operator whose config says `provider: openai-compatible`, with a server answering `answer` to its model list. */
async function serverDoctor(
  answer: (req: TransportRequest) => { status: number; body: string },
) {
  const { dir } = await fearTheTurtle(8);
  const deps = await operator();
  await mkdir(deps.configDir, { recursive: true });
  await writeFile(
    join(deps.configDir, "config.yaml"),
    `format: 1\nprovider: openai-compatible\nworkspaces:\n  fear-the-turtle: ${dir}\n`,
  );
  const requests: TransportRequest[] = [];
  deps.transport = {
    async request(req) {
      requests.push(req);
      const { status, body } = answer(req);
      return { status, body: new TextEncoder().encode(body) };
    },
  };
  deps.clipboard = noClipboard();
  const result = await doctor(deps);
  const check = result.checks.find((c) => c.subject === "LLM server");
  return { check, requests, result };
}

test("doctor asks a local server for its models and finds the configured one", async () => {
  const { check, requests } = await serverDoctor(() => ({
    status: 200,
    body: JSON.stringify({ data: [{ id: "qwen2.5:7b" }] }),
  }));
  expect(check).toMatchObject({ status: "ok" });
  expect((requests[0] as TransportRequest).url).toBe(`${OLLAMA_API}/models`);
});

test("doctor says to pull a model the server doesn't have, and lists what it has", async () => {
  const { check } = await serverDoctor(() => ({
    status: 200,
    body: JSON.stringify({ data: [{ id: "llama3.1:8b" }] }),
  }));
  expect(check?.status).toBe("warn");
  expect(check?.message).toContain("llama3.1:8b");
  expect(check?.fix).toContain("ollama pull qwen2.5:7b");
});

test("doctor warns, rather than fails, when the server isn't running", async () => {
  const { check, result } = await serverDoctor(() => {
    throw new Error("fetch failed");
  });
  expect(check?.status).toBe("warn");
  expect(check?.fix).toContain("ollama serve");
  expect(result.checks.some((c) => c.subject === "Anthropic key")).toBe(false);
});
