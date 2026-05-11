import numpy as np
import os

from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError
from ollama import AsyncClient
from dataclasses import dataclass

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
from ._utils import EmbeddingFunc


global_openai_async_client = None
global_azure_openai_async_client = None
global_ollama_client = None


def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client


def get_ollama_async_client_instance():
    global global_ollama_client
    if global_ollama_client is None:
        global_ollama_client = AsyncClient(host="http://127.0.0.1:11434")
    return global_ollama_client


@dataclass
class LLMConfig:
    embedding_func_raw: callable
    embedding_model_name: str
    embedding_dim: int
    embedding_max_token_size: int
    embedding_batch_num: int
    embedding_func_max_async: int
    query_better_than_threshold: float

    best_model_func_raw: callable
    best_model_name: str
    best_model_max_token_size: int
    best_model_max_async: int

    cheap_model_func_raw: callable
    cheap_model_name: str
    cheap_model_max_token_size: int
    cheap_model_max_async: int

    embedding_func: EmbeddingFunc = None
    best_model_func: callable = None
    cheap_model_func: callable = None

    def __post_init__(self):
        embedding_wrapper = wrap_embedding_func_with_attrs(
            embedding_dim=self.embedding_dim,
            max_token_size=self.embedding_max_token_size,
            model_name=self.embedding_model_name,
        )
        self.embedding_func = embedding_wrapper(self.embedding_func_raw)

        self.best_model_func = lambda prompt, *args, **kwargs: self.best_model_func_raw(
            self.best_model_name, prompt, *args, **kwargs
        )

        self.cheap_model_func = lambda prompt, *args, **kwargs: self.cheap_model_func_raw(
            self.cheap_model_name, prompt, *args, **kwargs
        )


# =========================
# OpenAI Configuration
# =========================

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    content = response.choices[0].message.content

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert({args_hash: {"return": content, "model": model}})
        await hashing_kv.index_done_callback()

    return content


async def gpt_4o_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model=model_name,
        input=texts,
        encoding_format="float",
    )
    return np.array([dp.embedding for dp in response.data])


openai_config = LLMConfig(
    embedding_func_raw=openai_embedding,
    embedding_model_name="text-embedding-3-small",
    embedding_dim=1536,
    embedding_max_token_size=8192,
    embedding_batch_num=32,
    embedding_func_max_async=1,
    query_better_than_threshold=0.2,

    best_model_func_raw=gpt_4o_complete,
    best_model_name="gpt-4o",
    best_model_max_token_size=32768,
    best_model_max_async=1,

    cheap_model_func_raw=gpt_4o_mini_complete,
    cheap_model_name="gpt-4o-mini",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=1,
)


openai_4o_mini_config = LLMConfig(
    embedding_func_raw=openai_embedding,
    embedding_model_name="text-embedding-3-small",
    embedding_dim=1536,
    embedding_max_token_size=8192,
    embedding_batch_num=32,
    embedding_func_max_async=1,
    query_better_than_threshold=0.2,

    best_model_func_raw=gpt_4o_mini_complete,
    best_model_name="gpt-4o-mini",
    best_model_max_token_size=32768,
    best_model_max_async=1,

    cheap_model_func_raw=gpt_4o_mini_complete,
    cheap_model_name="gpt-4o-mini",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=1,
)


# =========================
# Azure OpenAI Configuration
# =========================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        **kwargs,
    )

    content = response.choices[0].message.content

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert({args_hash: {"return": content, "model": deployment_name}})
        await hashing_kv.index_done_callback()

    return content


async def azure_gpt_4o_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model=model_name,
        input=texts,
        encoding_format="float",
    )
    return np.array([dp.embedding for dp in response.data])


azure_openai_config = LLMConfig(
    embedding_func_raw=azure_openai_embedding,
    embedding_model_name="text-embedding-3-small",
    embedding_dim=1536,
    embedding_max_token_size=8192,
    embedding_batch_num=32,
    embedding_func_max_async=1,
    query_better_than_threshold=0.2,

    best_model_func_raw=azure_gpt_4o_complete,
    best_model_name="gpt-4o",
    best_model_max_token_size=32768,
    best_model_max_async=1,

    cheap_model_func_raw=azure_gpt_4o_mini_complete,
    cheap_model_name="gpt-4o-mini",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=1,
)


# =========================
# Ollama Configuration
# =========================

async def ollama_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    ollama_client = get_ollama_async_client_instance()

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await ollama_client.chat(
        model=model,
        messages=messages,
    )

    content = response["message"]["content"]

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert({args_hash: {"return": content, "model": model}})
        await hashing_kv.index_done_callback()

    return content


async def ollama_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await ollama_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_mini_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await ollama_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def ollama_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    ollama_client = get_ollama_async_client_instance()

    response = await ollama_client.embed(
        model=model_name,
        input=texts,
    )

    embeddings = response["embeddings"]
    return np.array(embeddings)


ollama_config = LLMConfig(
    embedding_func_raw=ollama_embedding,
    embedding_model_name="nomic-embed-text",
    embedding_dim=768,
    embedding_max_token_size=8192,
    embedding_batch_num=1,
    embedding_func_max_async=1,
    query_better_than_threshold=0.2,

    best_model_func_raw=ollama_complete,
    best_model_name="qwen2.5:7b",
    best_model_max_token_size=32768,
    best_model_max_async=1,

    cheap_model_func_raw=ollama_mini_complete,
    cheap_model_name="qwen2.5:7b",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=1,
)


# =========================
# DeepSeek Configuration
# =========================

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def deepseek_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    import httpx

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY', 'sk-*******')}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 4096),
            },
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert({args_hash: {"return": content, "model": model}})
        await hashing_kv.index_done_callback()

    return content


async def deepseek_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await deepseek_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def bge_m3_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.siliconflow.cn/v1/embeddings",
            headers={
                "Authorization": f"Bearer {os.environ.get('SILICONFLOW_API_KEY', 'sk-******')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "BAAI/bge-m3",
                "input": texts,
                "encoding_format": "float",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        return np.array(embeddings)


deepseek_bge_config = LLMConfig(
    embedding_func_raw=bge_m3_embedding,
    embedding_model_name="BAAI/bge-m3",
    embedding_dim=1024,
    embedding_max_token_size=8192,
    embedding_batch_num=32,
    embedding_func_max_async=1,
    query_better_than_threshold=0.2,

    best_model_func_raw=deepseek_complete,
    best_model_name="deepseek-chat",
    best_model_max_token_size=32768,
    best_model_max_async=1,

    cheap_model_func_raw=deepseek_complete,
    cheap_model_name="deepseek-chat",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=1,
)
