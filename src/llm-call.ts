import { LlmError, type LlmProvider, type LlmReply } from "./anthropic.ts";
import {
  addSpend,
  baseUrlOf,
  type Config,
  ConfigFileError,
  configPath,
  modelOf,
  monthSpend,
  spendMonth,
} from "./config.ts";
import type { Deps } from "./deps.ts";
import { ceilingCost, costOf, priceOf, usd } from "./prices.ts";
import { type TranscriptPlace, writeTranscript } from "./transcripts.ts";

/** Room for a league-week's blurbs and intro, with the model's thinking. */
export const MAX_TOKENS = 16000;

/**
 * The reply ceiling for a provider other than Anthropic: a local model's
 * context is small, and a league-week's blurbs and intro take about 1,500 tokens.
 */
export const OTHER_MAX_TOKENS = 8000;

/**
 * The LLM stopped, or never ran, and the draft falls back. The message says
 * why, for the operator, without saying what the draft has instead.
 */
export class LlmStopped extends Error {
  override name = "LlmStopped";
}

export interface CallSetup {
  deps: Deps;
  key: string;
  config: Config;
  /** Where each call's LLM transcript goes. */
  transcript: TranscriptPlace;
}

/**
 * The `ask` a draft hands the LLM gate: each call sends one prompt and
 * returns the reply's text. Before each call, a model with no price, or a
 * ceiling cost that would pass the spend cap, stops it unmade. After it, its
 * LLM transcript is written and its reported tokens are added to the month's
 * spend. A call that can't be made or fails throws `LlmStopped`.
 */
export function llmAsk({
  deps,
  key,
  config,
  transcript,
}: CallSetup): (system: string, prompt: string) => Promise<string> {
  let calls = 0;
  return async (system, prompt) => {
    const price = priceOf(config);
    if (!price) {
      throw new LlmStopped(
        `${modelOf(config)} has no price in the price table, so it was never called: add it under prices in ${configPath(deps)}`,
      );
    }
    const baseUrl = baseUrlOf(config);
    const req = {
      key,
      provider: config.provider,
      ...(baseUrl === undefined ? {} : { baseUrl }),
      model: modelOf(config),
      system,
      prompt,
      maxTokens:
        config.provider === "anthropic" ? MAX_TOKENS : OTHER_MAX_TOKENS,
    };
    const month = spendMonth(deps.clock.now());
    await checkCap(deps, config, month, ceilingCost(price, req));
    const call = ++calls;
    const sent = { model: modelOf(config), system, prompt };
    let reply: LlmReply;
    try {
      reply = await deps.llm.write(req);
    } catch (err) {
      if (!(err instanceof LlmError)) throw err;
      // Spend first: a billed call must count even if its transcript can't be written.
      const cost = err.usage ? costOf(price, err.usage) : 0;
      if (err.usage) await stopOnBadFile(() => addSpend(deps, month, cost));
      await writeTranscript(transcript, call, {
        ...sent,
        ...(err.usage && { usage: err.usage }),
        cost,
        outcome: { error: err.message },
      });
      throw new LlmStopped(stopReason(err, config.provider));
    }
    const cost = costOf(price, reply.usage);
    await stopOnBadFile(() => addSpend(deps, month, cost));
    await writeTranscript(transcript, call, {
      ...sent,
      usage: reply.usage,
      cost,
      outcome: { reply: reply.text },
    });
    return reply.text;
  };
}

/** Throws `LlmStopped` when a call costing up to `ceiling` could pass the spend cap. */
async function checkCap(
  deps: Deps,
  config: Config,
  month: string,
  ceiling: number,
): Promise<void> {
  const spent = await stopOnBadFile(() => monthSpend(deps, month));
  const cap = config.spend_cap_usd;
  if (spent + ceiling <= cap) return;
  throw new LlmStopped(
    `The next LLM call could cost up to ${usd(ceiling)}, and ${usd(spent)} of this month's ${usd(cap)} spend cap is spent, so it wasn't made: raise the cap with lazy-pundit setup cap, or wait for next month`,
  );
}

/** Runs `f`, turning a `spend.yaml` that can't be read into `LlmStopped`. */
async function stopOnBadFile<T>(f: () => Promise<T>): Promise<T> {
  try {
    return await f();
  } catch (err) {
    if (err instanceof ConfigFileError) throw new LlmStopped(err.message);
    throw err;
  }
}

/** Why a failed call stops the LLM, with the fix where there is one. */
function stopReason(err: LlmError, provider: LlmProvider): string {
  if (err.kind !== "invalid-key") return err.message;
  if (provider === "huggingface") {
    return `Hugging Face refused the token (${err.message}): check HF_TOKEN or hf_token in credentials.yaml, and that the token may make calls to Inference Providers`;
  }
  if (provider === "openai-compatible") {
    return `The server refused the key (${err.message}): check LLM_API_KEY or llm_api_key in credentials.yaml`;
  }
  return `Anthropic refused the key (${err.message}): check it with \`lazy-pundit doctor\``;
}
