import time
import requests
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OllamaLLM:
    """Synchronous client for Ollama's generate API."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Check that Ollama is reachable and the model exists locally."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m.get("name") for m in resp.json().get("models", [])]
            
            if self.model not in models:
                logger.warning(
                    "Model %s not found in Ollama. Available: %s. It may be pulled automatically on first request.",
                    self.model, models,
                )
            else:
                logger.info("Connected to Ollama (%s), model: %s", self.base_url, self.model)
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama is running: ollama serve"
            )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from the LLM."""
        start = time.perf_counter()
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 1024,
            },
        }
        
        try:
            logger.debug("Base URL: %s", self.base_url)
            logger.debug("Generate URL: %s", f"{self.base_url}/api/generate")
            
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError("Lost connection to Ollama during generation.")
        except requests.Timeout:
            raise TimeoutError("Ollama generation timed out after 120s.")
            
        result = resp.json()
        answer = result.get("response", "").strip()
        elapsed = time.perf_counter() - start
        
        logger.info(
            "LLM response generated in %.2fs (%d tokens)",
            elapsed, result.get("eval_count", 0)
        )
        return answer
