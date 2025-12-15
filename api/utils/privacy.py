import re
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PrivacyManager:
    """
    Manager for handling privacy controls and user data protection.
    """

    def __init__(self, retention_days: int = 30):
        """
        Initialize privacy manager with data retention policy.
        """
        self.retention_days = retention_days

    def anonymize_user_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize user data by removing or obfuscating personal information.
        """
        anonymized_data = data.copy()

        # Remove or obfuscate personal identifiers
        personal_fields = [
            'user_id', 'email', 'name', 'phone', 'address',
            'ip_address', 'session_id', 'user_agent'
        ]

        for field in personal_fields:
            if field in anonymized_data:
                # For user_id and session_id, keep the first 3 characters and replace the rest with *
                if field in ['user_id', 'session_id'] and isinstance(anonymized_data[field], str):
                    if len(anonymized_data[field]) > 3:
                        anonymized_data[field] = anonymized_data[field][:3] + '*' * (len(anonymized_data[field]) - 3)
                    else:
                        anonymized_data[field] = '***'
                else:
                    anonymized_data[field] = '[REDACTED]'

        return anonymized_data

    def should_store_conversation(self, user_consent: bool = True) -> bool:
        """
        Determine if a conversation should be stored based on user consent.
        """
        return user_consent

    def sanitize_query(self, query: str) -> str:
        """
        Sanitize user query to remove potential PII.
        """
        # Remove email addresses
        query = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REMOVED]', query)

        # Remove potential phone numbers
        query = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REMOVED]', query)

        # Remove potential SSNs (simplified pattern)
        query = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REMOVED]', query)

        return query

    def get_retention_cutoff_date(self) -> datetime:
        """
        Get the cutoff date for data retention based on retention policy.
        """
        return datetime.now() - timedelta(days=self.retention_days)

    def should_delete_data(self, created_date: datetime) -> bool:
        """
        Determine if data should be deleted based on retention policy.
        """
        cutoff_date = self.get_retention_cutoff_date()
        return created_date < cutoff_date

    def encrypt_sensitive_data(self, data: str) -> str:
        """
        Simple encryption for sensitive data (in a real implementation, use proper encryption).
        This is a placeholder - use proper encryption in production.
        """
        # This is NOT real encryption - just obfuscation for demonstration
        # In a real implementation, use proper encryption like Fernet
        import base64
        encoded = base64.b64encode(data.encode()).decode()
        return f"encrypted:{encoded}"

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data (in a real implementation, use proper decryption).
        """
        if encrypted_data.startswith("encrypted:"):
            encoded = encrypted_data[10:]  # Remove "encrypted:" prefix
            try:
                return base64.b64decode(encoded).decode()
            except:
                return "[DECRYPTION_ERROR]"
        return encrypted_data

# Global privacy manager instance
privacy_manager = PrivacyManager()