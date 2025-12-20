"""
Configuration module for the Physical AI & Humanoid Robotics Book RAG Chatbot.
Manages application settings and configuration values with environment variable support.
"""

import os
from typing import Optional, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global configuration values dictionary
# This allows for runtime modification of config values
_config_values = {}


def get_config_value(key: str, default_value: Optional[Union[str, int, float, bool]] = None) -> Union[str, int, float, bool]:
    """
    Get a configuration value from environment variables or defaults.

    This function checks in the following order:
    1. Runtime-modified config values (from _config_values dict)
    2. Environment variables
    3. Default value provided as parameter
    4. Default values defined in this module

    Args:
        key: Configuration key to look up
        default_value: Default value to return if key is not found

    Returns:
        Configuration value, with automatic type conversion for numbers and booleans
    """
    # First, check if the value was set at runtime
    if key in _config_values:
        value = _config_values[key]
        return _convert_value_type(value)

    # Second, check environment variables
    env_value = os.getenv(key)
    if env_value is not None:
        return _convert_value_type(env_value)

    # Third, check if a default was provided as parameter
    if default_value is not None:
        # Convert default_value to string before passing to _convert_value_type
        # This handles the case where default_value is already an int, float, or bool
        return _convert_value_type(str(default_value))

    # Fourth, check for module-defined defaults
    module_defaults = {
        # Database configuration (optional - system works without database)
        'DATABASE_URL': 'postgresql://user:password@localhost:5432/book_rag_chatbot',  # Optional: for question logging

        # Qdrant vector database configuration
        'QDRANT_URL': 'http://localhost:6333',  # Default to local; override with env var for cloud
        'QDRANT_API_KEY': None,  # Add support for API key
        'QDRANT_COLLECTION_NAME': 'book_chunks',

        # OpenAI configuration
        'EMBEDDING_MODEL': 'text-embedding-3-small',
        'CHAT_MODEL': 'gpt-4o',

        # Ingestion pipeline configuration
        'MAX_CHUNK_SIZE': '1000',
        'CHUNK_OVERLAP': '100',
        'DEFAULT_TOP_K': '5',
        'SEARCH_THRESHOLD': '0.3',

        # Cache configuration
        'CACHE_TTL_SECONDS': '3600',
        'CACHE_MAX_MEMORY_ITEMS': '1000',
        'REDIS_URL': 'redis://localhost:6379',

        # Rate limiting configuration
        'RATE_LIMIT_REQUESTS': '100',
        'RATE_LIMIT_WINDOW': '3600',

        # API configuration
        'API_KEY': 'your-default-api-key',
        'API_KEYS': '',

        # Frontend configuration
        'FRONTEND_API_URL': 'http://localhost:8000',
        'DEFAULT_BOOK_ID': 'default-book',

        # Logging configuration
        'LOG_LEVEL': 'INFO',
        'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',

        # Security configuration
        'JWT_SECRET_KEY': 'your-secret-key-change-in-production',
        'JWT_ALGORITHM': 'HS256',
        'JWT_EXPIRATION_HOURS': '24',

        # CORS configuration
        'ALLOWED_ORIGINS': '*',

        # Processing configuration
        'MAX_QUESTION_LENGTH': '2000',
        'MAX_SELECTED_TEXT_LENGTH': '5000',
        'MAX_CONTEXT_LENGTH': '10000',

        # Timeout configuration
        'OPENAI_API_TIMEOUT': '30',
        'VECTOR_DB_TIMEOUT': '10',
    }

    if key in module_defaults:
        return _convert_value_type(module_defaults[key])

    # If no value is found anywhere, return the default (which may be None)
    return _convert_value_type(str(default_value)) if default_value is not None else None


def set_config_value(key: str, value: Union[str, int, float, bool]):
    """
    Set a configuration value at runtime.

    Args:
        key: Configuration key to set
        value: Value to set for the key
    """
    _config_values[key] = str(value)


def _convert_value_type(value: str) -> Union[str, int, float, bool]:
    """
    Convert a string value to the appropriate Python type.

    Args:
        value: String value to convert

    Returns:
        Value converted to appropriate type (str, int, float, or bool)
    """
    if value is None:
        return None

    # Check for boolean values
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'

    # Check for integer values
    try:
        if '.' not in value and 'e' not in value.lower():
            return int(value)
    except ValueError:
        pass

    # Check for float values
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string if no other type matches
    return value


def get_bool_config(key: str, default_value: bool = False) -> bool:
    """
    Get a boolean configuration value.

    Args:
        key: Configuration key to look up
        default_value: Default boolean value to return if key is not found

    Returns:
        Boolean configuration value
    """
    value = get_config_value(key, str(default_value))
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    elif isinstance(value, int):
        return bool(value)
    return default_value


def get_int_config(key: str, default_value: int = 0) -> int:
    """
    Get an integer configuration value.

    Args:
        key: Configuration key to look up
        default_value: Default integer value to return if key is not found

    Returns:
        Integer configuration value
    """
    value = get_config_value(key, str(default_value))
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        return int(value)
    elif isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default_value
    return default_value


def get_float_config(key: str, default_value: float = 0.0) -> float:
    """
    Get a float configuration value.

    Args:
        key: Configuration key to look up
        default_value: Default float value to return if key is not found

    Returns:
        Float configuration value
    """
    value = get_config_value(key, str(default_value))
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default_value
    return default_value


def get_list_config(key: str, default_value: list = None, separator: str = ',') -> list:
    """
    Get a list configuration value (typically from comma-separated string).

    Args:
        key: Configuration key to look up
        default_value: Default list value to return if key is not found
        separator: Separator used to split the string (default: comma)

    Returns:
        List configuration value
    """
    if default_value is None:
        default_value = []

    value = get_config_value(key)
    if isinstance(value, list):
        return value
    elif isinstance(value, str):
        if value.strip():
            return [item.strip() for item in value.split(separator)]
        else:
            return []
    return default_value


def reset_config():
    """
    Reset runtime configuration values to empty.
    This does not affect environment variables or module defaults.
    """
    _config_values.clear()


def get_all_config_values() -> dict:
    """
    Get all current configuration values.

    Returns:
        Dictionary containing all configuration values
    """
    all_values = {}

    # Add runtime config values
    all_values.update(_config_values)

    # Add environment variables that start with common prefixes
    for key, value in os.environ.items():
        if key.startswith(('API_', 'DB_', 'QDRANT_', 'CACHE_', 'RATE_LIMIT_', 'FRONTEND_')):
            if key not in all_values:  # Don't overwrite runtime values
                all_values[key] = value

    return all_values


def validate_config() -> list:
    """
    Validate the configuration and return a list of warnings.

    Returns:
        List of validation warning messages
    """
    warnings = []

    # Check if required environment variables are set with default values
    required_keys = ['DATABASE_URL', 'QDRANT_URL', 'API_KEY']

    for key in required_keys:
        value = get_config_value(key)
        if value and 'your-' in value.lower() and 'change' in value.lower():
            warnings.append(f"Configuration '{key}' contains default placeholder value: {value}")

    # Check for potential security issues
    api_key = get_config_value('API_KEY')
    if api_key and len(api_key) < 20:
        warnings.append("API_KEY is shorter than recommended (should be at least 20 characters)")

    jwt_secret = get_config_value('JWT_SECRET_KEY')
    if jwt_secret and jwt_secret == 'your-secret-key-change-in-production':
        warnings.append("JWT_SECRET_KEY is using default insecure value")

    return warnings


# Initialize any necessary config values on module import
def _init_config():
    """Initialize configuration module."""
    # This function runs when the module is imported
    pass


# Run initialization
_init_config()


if __name__ == "__main__":
    # Example usage
    print("Configuration module loaded.")
    print(f"Database URL: {get_config_value('DATABASE_URL')}")
    print(f"Qdrant URL: {get_config_value('QDRANT_URL')}")
    print(f"Max chunk size: {get_config_value('MAX_CHUNK_SIZE')}")

    # Show validation warnings
    warnings = validate_config()
    if warnings:
        print("\nConfiguration warnings:")
        for warning in warnings:
            print(f"  - {warning}")