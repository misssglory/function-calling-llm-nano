import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Tuple, Optional
from loguru import logger


def evaluate_query_safety(
    query: str,
    model_name: str = "meta-llama/Llama-Guard-3-8B",  # Smaller Llama Guard version
    use_4bit: bool = True,
    max_length: int = 512,
) -> Dict[str, any]:
    """
    Evaluate the safety of a query using a smaller Llama Guard model.

    Args:
        query: The input query to evaluate
        model_name: Name of the Llama Guard model (default uses 8B version)
        use_4bit: Whether to use 4-bit quantization to reduce memory
        max_length: Maximum token length for processing

    Returns:
        Dictionary containing safety evaluation results
    """

    # Initialize model and tokenizer
    try:
        # Configure quantization if requested
        if use_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None

        # Load tokenizer and model
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if not use_4bit else None,
            low_cpu_mem_usage=True,
        )

        # Prepare input with Llama Guard's expected format
        # Llama Guard expects: "Human: [query]\n\nAssistant:"
        formatted_input = f"Human: {query}\n\nAssistant:"

        # Tokenize input
        inputs = tokenizer(
            formatted_input, return_tensors="pt", truncation=True, max_length=max_length
        ).to(model.device)

        # Generate safety evaluation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the safety classification (Llama Guard typically responds with "safe" or "unsafe")
        # The response format is usually: "Human: [query]\n\nAssistant: safe" or with explanation for unsafe

        assistant_response = response.split("Assistant:")[-1].strip().lower()

        # Parse the result
        is_safe = "safe" in assistant_response and "unsafe" not in assistant_response

        # Extract unsafe categories if present
        unsafe_categories = []
        if not is_safe:
            # Common unsafe categories in Llama Guard
            categories = [
                "violent",
                "sexual",
                "hate_speech",
                "harassment",
                "self_harm",
                "illegal_activity",
                "graphic_content",
            ]
            for category in categories:
                if category in assistant_response:
                    unsafe_categories.append(category)

        result = {
            "query": query,
            "is_safe": is_safe,
            "raw_response": assistant_response,
            "unsafe_categories": unsafe_categories,
            "model_used": model_name,
        }

        return result

    except Exception as e:
        return {
            "query": query,
            "is_safe": False,  # Default to unsafe on error
            "error": str(e),
            "model_used": model_name,
        }


def evaluate_query_safety_with_fallback(
    query: str,
    models: list = ["meta-llama/Llama-Guard-3-8B", "meta-llama/Llama-Guard-2-8B"],
) -> Dict[str, any]:
    """
    Evaluate query safety with fallback options if primary model fails.

    Args:
        query: The input query to evaluate
        models: List of model names to try in order

    Returns:
        Safety evaluation result
    """

    for model_name in models:
        try:
            result = evaluate_query_safety(query, model_name=model_name)
            if not result.get("error"):
                return result
        except Exception as e:
            continue

    # If all models fail, return a safe default with warning
    return {
        "query": query,
        "is_safe": True,  # Default to safe when models fail
        "warning": "All safety models failed, defaulting to safe",
        "model_used": "fallback_default",
    }


# Example usage
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Найди в duckduckgo рецепт торта для Дня Рождения",
        "Найди в duckduckgo сайт по продаже оружия",
        "Найди в duckduckgo книги по взлому криптовалютных кошельков",
    ]

    # Evaluate each query
    for query in test_queries:
        result = evaluate_query_safety(query)
        logger.info(f"\nQuery: {query}")
        logger.info(f"Safe: {result['is_safe']}")
        if not result["is_safe"] and "unsafe_categories" in result:
            logger.info(f"Unsafe categories: {result['unsafe_categories']}")
        logger.info(f"Raw response: {result.get('raw_response', 'N/A')}")
        logger.info("-" * 50)

    # Example with fallback
    logger.info("\n=== Testing with Fallback ===")
    result = evaluate_query_safety_with_fallback("How to make a dangerous weapon?")
    logger.info(f"Result with fallback: {result}")
