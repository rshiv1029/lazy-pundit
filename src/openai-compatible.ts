import { z } from "zod";
import {
  anthropicLlm,
  type Llm,
  LlmError,
  type LlmErrorKind,
  type LlmReply,
} from "./anthropic.ts";
import type { Transport } from "./transport.ts";

/** Hugging Face's Inference Providers router: an OpenAI-compatible chat API. */
export const HUGGING_FACE_API = "https://router.huggingface.co/v1";

/** The model a `provider: huggingface` config uses when it names none. */
export const HUGGING_FACE_DEFAULT_MODEL = "openai/gpt-oss-120b";

/** Ollama's OpenAI-compatible API on its default port: where `provider: openai-compatible` points when it names none. */
export const OLLAMA_API = "http://localhost:11434/v1";

/** The model a `provider: openai-compatible` config uses when it names none: an Ollama tag. */
export const OLLAMA_DEFAULT_MODEL = "qwen2.5:7b";

/** One chunk of a streamed reply; the last one carries the token counts. */
const chunkSchema = z.object({
  choices: z
    .array(
      z.object({
        delta: z.object({ content: z.string().nullish() }).optional(),
        finish_reason: z.string().nullish(),
      }),
    )
    .default([]),
  usage: z
    .object({ prompt_tokens: z.number(), completion_tokens: z.number() })
    .nullish(),
  error: z.unknown().optional(),
});

/** A whole reply, from a server that ignored `stream`. */
const completionSchema = z.object({
  choices: z
    .array(
      z.object({
        message: z.object({ content: z.string().nullable() }),
        finish_reason: z.string().nullable(),
      }),
    )
    .min(1),
  usage: z
    .object({ prompt_tokens: z.number(), completion_tokens: z.number() })
    .nullish(),
});

interface Parsed {
  content: string;
  finish: string | undefined;
  usage: { inputTokens: number; outputTokens: number };
}

/**
 * Chat completions from any OpenAI-compatible server (Hugging Face's router,
 * Ollama, OpenRouter, Groq) through the transport. The reply is asked for
 * streamed, so a slow local model, which sends its first bytes at once and
 * the rest as it writes, isn't cut off by the wait for a whole answer.
 * Throws an `LlmError` on an error answer, a network failure, and a reply that
 * doesn't end with `stop`, so a half-written blurb never reaches a draft.
 */
export function openAiCompatibleLlm(transport: Transport): Llm {
  return {
    async write(req): Promise<LlmReply> {
      const base = (req.baseUrl ?? OLLAMA_API).replace(/\/+$/, "");
      const url = `${base}/chat/completions`;
      const name = serverName(base);
      let res: Awaited<ReturnType<Transport["request"]>>;
      try {
        res = await transport.request({
          url,
          method: "POST",
          headers: {
            "content-type": "application/json",
            ...(req.key ? { authorization: `Bearer ${req.key}` } : {}),
          },
          body: JSON.stringify({
            model: req.model,
            max_tokens: req.maxTokens,
            stream: true,
            stream_options: { include_usage: true },
            messages: [
              { role: "system", content: req.system },
              { role: "user", content: req.prompt },
            ],
          }),
        });
      } catch (err) {
        throw new LlmError(
          "provider",
          `Couldn't reach ${name}: ${(err as Error).message}${localHint(base, `Is it running? For Ollama, start it and run: ollama pull ${req.model}`)}`,
        );
      }
      const text = new TextDecoder().decode(res.body);
      if (res.status !== 200) {
        const detail = errorMessage(text);
        const hint =
          res.status === 404
            ? localHint(base, `For Ollama, run: ollama pull ${req.model}`)
            : "";
        throw new LlmError(
          errorKind(res.status),
          `${name} answered ${res.status}${detail ? `: ${detail}` : ""}${hint}`,
        );
      }
      const parsed = parseReply(text);
      if (parsed === undefined) {
        throw new LlmError(
          "provider",
          `${url} answered with something that isn't a chat completion; check that nothing on this network intercepts requests to ${name}`,
        );
      }
      if (typeof parsed === "string") {
        throw new LlmError("provider", `${name} reported an error: ${parsed}`);
      }
      if (parsed.finish !== "stop") {
        throw new LlmError(
          "provider",
          `${name}'s reply stopped early (${parsed.finish ?? "the connection closed"}), so it wasn't used`,
          parsed.usage,
        );
      }
      return {
        // Some reasoning models put their thinking in the reply.
        text: parsed.content.replace(/<think>[\s\S]*?<\/think>/g, "").trim(),
        usage: parsed.usage,
      };
    },
  };
}

/** The LLM a draft calls: Anthropic's for `anthropic`, else an OpenAI-compatible server's. */
export function routedLlm(transport: Transport): Llm {
  const anthropic = anthropicLlm(transport);
  const compatible = openAiCompatibleLlm(transport);
  return {
    write: (req) =>
      (req.provider === undefined || req.provider === "anthropic"
        ? anthropic
        : compatible
      ).write(req),
  };
}

/**
 * The text and token counts of a streamed reply (`data: {...}` lines) or of a
 * whole one; a string for an error the server sent mid-stream; undefined for
 * anything else.
 */
function parseReply(text: string): Parsed | string | undefined {
  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.startsWith("data:"))
    .map((l) => l.slice(5).trim())
    .filter((l) => l !== "" && l !== "[DONE]");
  if (lines.length === 0) {
    const whole = completionSchema.safeParse(parseJson(text));
    const choice = whole.success ? whole.data.choices[0] : undefined;
    if (!whole.success || !choice) return undefined;
    return {
      content: choice.message.content ?? "",
      finish: choice.finish_reason ?? undefined,
      usage: {
        inputTokens: whole.data.usage?.prompt_tokens ?? 0,
        outputTokens: whole.data.usage?.completion_tokens ?? 0,
      },
    };
  }
  const parsed: Parsed = {
    content: "",
    finish: undefined,
    usage: { inputTokens: 0, outputTokens: 0 },
  };
  for (const line of lines) {
    const chunk = chunkSchema.safeParse(parseJson(line));
    if (!chunk.success) return undefined;
    if (chunk.data.error !== undefined) {
      return errorText(chunk.data.error);
    }
    for (const choice of chunk.data.choices) {
      parsed.content += choice.delta?.content ?? "";
      if (choice.finish_reason) parsed.finish = choice.finish_reason;
    }
    if (chunk.data.usage) {
      parsed.usage = {
        inputTokens: chunk.data.usage.prompt_tokens,
        outputTokens: chunk.data.usage.completion_tokens,
      };
    }
  }
  return parsed;
}

/** What to call the server in a message: Hugging Face, else its host. */
function serverName(base: string): string {
  if (base === HUGGING_FACE_API) return "Hugging Face";
  try {
    return new URL(base).host;
  } catch {
    return base;
  }
}

/** `hint` after a space, for a server on this machine; otherwise nothing. */
function localHint(base: string, hint: string): string {
  return /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])[:/]/.test(base)
    ? ` ${hint}`
    : "";
}

/** 401 and 403 are a refused key; 402 is the credits used up. */
function errorKind(status: number): LlmErrorKind {
  if (status === 401 || status === 403) return "invalid-key";
  if (status === 402) return "no-credits";
  return "provider";
}

const errorSchema = z.union([
  z.object({ error: z.string() }),
  z.object({ error: z.object({ message: z.string() }) }),
]);

/** The message in an error body, or "". */
function errorMessage(body: string): string {
  const parsed = errorSchema.safeParse(parseJson(body));
  if (!parsed.success) return "";
  return errorText(parsed.data.error);
}

function errorText(error: unknown): string {
  if (typeof error === "string") return error;
  const message = (error as { message?: unknown } | null)?.message;
  return typeof message === "string" ? message : "unknown error";
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}
