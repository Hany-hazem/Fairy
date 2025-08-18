"""
Privacy and Security Manager

This module handles privacy controls, security measures, permission management,
and user consent tracking for the personal assistant.
"""

import json
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pathlib import Path
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .personal_assistant_models import UserPermission, PermissionType

logger = logging.getLogger(__name__)


class ConsentStatus(Enum):
    """Status of user consent"""
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    REVOKED = "revoked"
    EXPIRED = "expired"


class DataCategory(Enum):
    """Categories of personal data"""
    PERSONAL_INFO = "personal_info"
    INTERACTION_HISTORY = "interaction_history"
    FILE_ACCESS = "file_access"
    SCREEN_CONTENT = "screen_content"
    PREFERENCES = "preferences"
    LEARNING_DATA = "learning_data"
    SYSTEM_LOGS = "system_logs"
    BIOMETRIC_DATA = "biometric_data"
    LOCATION_DATA = "location_data"
    COMMUNICATION_DATA = "communication_data"


class PrivacyLevel(Enum):
    """Privacy levels for different data types"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class DataRetentionPolicy(Enum):
    """Data retention policies"""
    SESSION_ONLY = "session_only"
    SHORT_TERM = "short_term"  # 30 days
    MEDIUM_TERM = "medium_term"  # 1 year
    LONG_TERM = "long_term"  # 5 years
    PERMANENT = "permanent"


class PrivacySecurityManager:
    """Manages privacy controls, security, and user permissions"""
    
    def __init__(self, db_path: str = "personal_assistant.db", encryption_key: Optional[bytes] = None):
        self.db_path = db_path
        self._init_database()
        self._init_encryption(encryption_key)
        self._permission_cache: Dict[str, Dict[PermissionType, bool]] = {}
        self._consent_cache: Dict[str, Dict[DataCategory, ConsentStatus]] = {}
        self._privacy_settings_cache: Dict[str, Dict[str, Any]] = {}
        self._data_retention_policies: Dict[DataCategory, DataRetentionPolicy] = self._init_default_retention_policies()
        self._privacy_change_listeners: List[callable] = []
    
    def get_connection(self):
        """Get a database connection"""
        return sqlite3.connect(self.db_path)
    
    def _init_database(self):
        """Initialize the database schema for privacy and security"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    permission_type TEXT NOT NULL,
                    granted BOOLEAN DEFAULT FALSE,
                    granted_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    scope TEXT,
                    revoked BOOLEAN DEFAULT FALSE,
                    revoked_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, permission_type)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_consent (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    data_category TEXT NOT NULL,
                    consent_status TEXT NOT NULL,
                    granted_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    revoked_at TIMESTAMP,
                    consent_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, data_category)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS encrypted_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    data_key TEXT NOT NULL,
                    encrypted_content BLOB NOT NULL,
                    data_category TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, data_key)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_deletion_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    data_categories TEXT NOT NULL,
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    details TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS privacy_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    setting_key TEXT NOT NULL,
                    setting_value TEXT NOT NULL,
                    privacy_level TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, setting_key)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_retention_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    data_category TEXT NOT NULL,
                    retention_policy TEXT NOT NULL,
                    custom_days INTEGER,
                    auto_delete BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, data_category)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    data_category TEXT NOT NULL,
                    access_type TEXT NOT NULL,
                    data_key TEXT,
                    purpose TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS privacy_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolution_details TEXT
                )
            """)
            
            conn.commit()
    
    def _init_encryption(self, encryption_key: Optional[bytes] = None):
        """Initialize encryption for sensitive data"""
        if encryption_key:
            self.encryption_key = encryption_key
        else:
            # Generate or load encryption key
            key_file = Path("encryption.key")
            if key_file.exists():
                with open(key_file, "rb") as f:
                    self.encryption_key = f.read()
            else:
                self.encryption_key = Fernet.generate_key()
                with open(key_file, "wb") as f:
                    f.write(self.encryption_key)
        
        self.cipher_suite = Fernet(self.encryption_key)
    
    async def request_permission(self, user_id: str, permission_type: PermissionType, 
                               scope: Optional[Dict[str, Any]] = None,
                               expires_in_days: Optional[int] = None) -> bool:
        """Request permission from user for specific data access"""
        # Check if permission already exists and is valid
        existing_permission = await self.get_permission(user_id, permission_type)
        if existing_permission and existing_permission.granted and not existing_permission.revoked:
            if not existing_permission.expires_at or existing_permission.expires_at > datetime.now():
                return True
        
        # Log permission request
        await self._log_audit_event(user_id, "permission_requested", 
                                   f"{permission_type.value}", 
                                   {"scope": scope})
        
        # In a real implementation, this would trigger a UI prompt
        # For now, we'll simulate user approval based on permission type
        granted = await self._simulate_user_permission_response(permission_type)
        
        if granted:
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now() + timedelta(days=expires_in_days)
            
            await self._grant_permission(user_id, permission_type, scope, expires_at)
        else:
            await self._deny_permission(user_id, permission_type)
        
        return granted
    
    async def get_permission(self, user_id: str, permission_type: PermissionType) -> Optional[UserPermission]:
        """Get current permission status for a user and permission type"""
        # Check cache first
        if user_id in self._permission_cache and permission_type in self._permission_cache[user_id]:
            # Verify cache is still valid by checking database
            pass
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT granted, granted_at, expires_at, scope, revoked, revoked_at
                FROM user_permissions 
                WHERE user_id = ? AND permission_type = ?
            """, (user_id, permission_type.value))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            permission = UserPermission(
                user_id=user_id,
                permission_type=permission_type,
                granted=row[0],
                granted_at=datetime.fromisoformat(row[1]) if row[1] else None,
                expires_at=datetime.fromisoformat(row[2]) if row[2] else None,
                scope=json.loads(row[3]) if row[3] else {},
                revoked=row[4],
                revoked_at=datetime.fromisoformat(row[5]) if row[5] else None
            )
            
            # Update cache
            if user_id not in self._permission_cache:
                self._permission_cache[user_id] = {}
            self._permission_cache[user_id][permission_type] = permission.granted and not permission.revoked
            
            return permission
    
    async def revoke_permission(self, user_id: str, permission_type: PermissionType) -> bool:
        """Revoke a previously granted permission"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE user_permissions 
                SET revoked = TRUE, revoked_at = ?
                WHERE user_id = ? AND permission_type = ?
            """, (datetime.now(), user_id, permission_type.value))
            
            if cursor.rowcount > 0:
                conn.commit()
                
                # Update cache
                if user_id in self._permission_cache and permission_type in self._permission_cache[user_id]:
                    self._permission_cache[user_id][permission_type] = False
                
                # Log revocation
                await self._log_audit_event(user_id, "permission_revoked", 
                                           f"{permission_type.value}")
                return True
        
        return False
    
    async def check_permission(self, user_id: str, permission_type: PermissionType) -> bool:
        """Check if user has granted permission for specific access"""
        permission = await self.get_permission(user_id, permission_type)
        if not permission:
            return False
        
        # Check if permission is valid
        if not permission.granted or permission.revoked:
            return False
        
        # Check if permission has expired
        if permission.expires_at and permission.expires_at < datetime.now():
            await self._expire_permission(user_id, permission_type)
            return False
        
        return True
    
    async def request_consent(self, user_id: str, data_category: DataCategory,
                            purpose: str, retention_days: Optional[int] = None) -> ConsentStatus:
        """Request user consent for data collection/processing"""
        # Check existing consent
        existing_consent = await self.get_consent_status(user_id, data_category)
        if existing_consent in [ConsentStatus.GRANTED]:
            return existing_consent
        
        # Log consent request
        await self._log_audit_event(user_id, "consent_requested", 
                                   f"{data_category.value}", 
                                   {"purpose": purpose, "retention_days": retention_days})
        
        # Simulate user consent response
        consent_status = await self._simulate_user_consent_response(data_category)
        
        expires_at = None
        if retention_days and consent_status == ConsentStatus.GRANTED:
            expires_at = datetime.now() + timedelta(days=retention_days)
        
        await self._record_consent(user_id, data_category, consent_status, 
                                 {"purpose": purpose}, expires_at)
        
        return consent_status
    
    async def get_consent_status(self, user_id: str, data_category: DataCategory) -> ConsentStatus:
        """Get current consent status for a data category"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT consent_status, expires_at, revoked_at
                FROM user_consent 
                WHERE user_id = ? AND data_category = ?
                ORDER BY updated_at DESC
                LIMIT 1
            """, (user_id, data_category.value))
            row = cursor.fetchone()
            
            if not row:
                return ConsentStatus.PENDING
            
            status = ConsentStatus(row[0])
            expires_at = datetime.fromisoformat(row[1]) if row[1] else None
            revoked_at = datetime.fromisoformat(row[2]) if row[2] else None
            
            # Check if consent has expired or been revoked
            if revoked_at:
                return ConsentStatus.REVOKED
            if expires_at and expires_at < datetime.now():
                await self._expire_consent(user_id, data_category)
                return ConsentStatus.EXPIRED
            
            return status
    
    async def revoke_consent(self, user_id: str, data_category: DataCategory) -> bool:
        """Revoke previously granted consent"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE user_consent 
                SET consent_status = ?, revoked_at = ?, updated_at = ?
                WHERE user_id = ? AND data_category = ?
            """, (ConsentStatus.REVOKED.value, datetime.now(), datetime.now(),
                  user_id, data_category.value))
            
            if cursor.rowcount > 0:
                conn.commit()
                
                # Update cache
                if user_id in self._consent_cache:
                    self._consent_cache[user_id][data_category] = ConsentStatus.REVOKED
                
                # Log revocation
                await self._log_audit_event(user_id, "consent_revoked", 
                                           f"{data_category.value}")
                return True
        
        return False
    
    async def encrypt_personal_data(self, user_id: str, data_key: str, 
                                  data: Dict[str, Any], data_category: DataCategory,
                                  privacy_level: PrivacyLevel = PrivacyLevel.CONFIDENTIAL) -> bool:
        """Encrypt and store personal data with enhanced security"""
        try:
            # Check if user has consented to data storage
            consent_status = await self.get_consent_status(user_id, data_category)
            if consent_status != ConsentStatus.GRANTED:
                await self.report_privacy_violation(
                    user_id, "unauthorized_data_storage",
                    f"Attempted to store {data_category.value} data without consent",
                    "high"
                )
                return False
            
            # Add metadata for enhanced tracking
            enhanced_data = {
                "content": data,
                "privacy_level": privacy_level.value,
                "encrypted_at": datetime.now().isoformat(),
                "retention_policy": self._data_retention_policies.get(data_category, DataRetentionPolicy.MEDIUM_TERM).value
            }
            
            # Serialize and encrypt data
            data_json = json.dumps(enhanced_data)
            encrypted_data = self.cipher_suite.encrypt(data_json.encode())
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO encrypted_data 
                    (user_id, data_key, encrypted_content, data_category, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, data_key, encrypted_data, data_category.value, datetime.now()))
                conn.commit()
            
            # Log data access for transparency
            await self.log_data_access(user_id, data_category, "write", data_key, "data_storage")
            await self._log_audit_event(user_id, "data_encrypted", data_key, 
                                       {"privacy_level": privacy_level.value})
            return True
            
        except Exception as e:
            logger.error(f"Failed to encrypt data for user {user_id}: {e}")
            await self.report_privacy_violation(
                user_id, "encryption_failure",
                f"Failed to encrypt {data_category.value} data: {str(e)}",
                "high"
            )
            return False
    
    async def decrypt_personal_data(self, user_id: str, data_key: str, 
                                  purpose: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Decrypt and retrieve personal data with enhanced access control"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT encrypted_content, data_category FROM encrypted_data 
                    WHERE user_id = ? AND data_key = ?
                """, (user_id, data_key))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                data_category = DataCategory(row[1])
                
                # Check if user still has consent for this data access
                consent_status = await self.get_consent_status(user_id, data_category)
                if consent_status not in [ConsentStatus.GRANTED]:
                    await self.report_privacy_violation(
                        user_id, "unauthorized_data_access",
                        f"Attempted to access {data_category.value} data without valid consent",
                        "high"
                    )
                    return None
                
                # Decrypt data
                decrypted_data = self.cipher_suite.decrypt(row[0])
                enhanced_data = json.loads(decrypted_data.decode())
                
                # Log data access for transparency
                await self.log_data_access(user_id, data_category, "read", data_key, purpose)
                await self._log_audit_event(user_id, "data_decrypted", data_key, 
                                           {"purpose": purpose})
                
                # Return just the content if it's enhanced format, otherwise return as-is
                if isinstance(enhanced_data, dict) and "content" in enhanced_data:
                    return enhanced_data["content"]
                return enhanced_data
                
        except Exception as e:
            logger.error(f"Failed to decrypt data for user {user_id}: {e}")
            await self.report_privacy_violation(
                user_id, "decryption_failure",
                f"Failed to decrypt data: {str(e)}",
                "medium"
            )
            return None
    
    async def request_data_deletion(self, user_id: str, data_categories: List[DataCategory],
                                  reason: Optional[str] = None) -> str:
        """Request deletion of user data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO data_deletion_requests 
                (user_id, data_categories, details)
                VALUES (?, ?, ?)
            """, (user_id, 
                  json.dumps([cat.value for cat in data_categories]),
                  json.dumps({"reason": reason})))
            
            # Get the auto-generated ID
            cursor = conn.execute("SELECT last_insert_rowid()")
            request_id = str(cursor.fetchone()[0])
            conn.commit()
        
        await self._log_audit_event(user_id, "data_deletion_requested", 
                                   f"categories: {[cat.value for cat in data_categories]}")
        
        # Process deletion immediately for demo
        await self._process_data_deletion(request_id)
        
        return request_id
    
    async def _process_data_deletion(self, request_id: str) -> bool:
        """Process a data deletion request"""
        with sqlite3.connect(self.db_path) as conn:
            # Get deletion request details
            cursor = conn.execute("""
                SELECT user_id, data_categories FROM data_deletion_requests 
                WHERE id = ? AND status = 'pending'
            """, (request_id,))
            row = cursor.fetchone()
            
            if not row:
                return False
            
            user_id, categories_json = row
            categories = [DataCategory(cat) for cat in json.loads(categories_json)]
            
            # Delete data based on categories
            for category in categories:
                await self._delete_data_by_category(user_id, category)
            
            # Mark request as processed
            conn.execute("""
                UPDATE data_deletion_requests 
                SET status = 'completed', processed_at = ?
                WHERE id = ?
            """, (datetime.now(), request_id))
            conn.commit()
            
            await self._log_audit_event(user_id, "data_deletion_completed", request_id)
            return True
    
    async def _delete_data_by_category(self, user_id: str, category: DataCategory) -> None:
        """Delete user data by category with comprehensive cleanup"""
        with sqlite3.connect(self.db_path) as conn:
            # Delete encrypted data for this category
            conn.execute("DELETE FROM encrypted_data WHERE user_id = ? AND data_category = ?", 
                        (user_id, category.value))
            
            # Category-specific deletions - only delete from tables that exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            if category == DataCategory.INTERACTION_HISTORY:
                if "user_interactions" in existing_tables:
                    conn.execute("DELETE FROM user_interactions WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM data_access_log WHERE user_id = ? AND data_category = ?", 
                           (user_id, category.value))
            elif category == DataCategory.PERSONAL_INFO:
                if "user_contexts" in existing_tables:
                    conn.execute("DELETE FROM user_contexts WHERE user_id = ?", (user_id,))
                # Don't delete all privacy settings for personal info deletion
            elif category == DataCategory.PREFERENCES:
                conn.execute("DELETE FROM privacy_settings WHERE user_id = ?", (user_id,))
                # Clear cache
                if user_id in self._privacy_settings_cache:
                    del self._privacy_settings_cache[user_id]
            elif category == DataCategory.SYSTEM_LOGS:
                conn.execute("DELETE FROM audit_log WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM data_access_log WHERE user_id = ?", (user_id,))
            
            # Always clean up related access logs for this category
            conn.execute("DELETE FROM data_access_log WHERE user_id = ? AND data_category = ?", 
                        (user_id, category.value))
            
            conn.commit()
            
            # Clear relevant caches
            if user_id in self._permission_cache:
                # Only clear permissions related to this category
                pass  # Keep permissions for other categories
            if user_id in self._consent_cache:
                if category in self._consent_cache[user_id]:
                    del self._consent_cache[user_id][category]
    
    async def get_privacy_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """Get privacy dashboard data for user"""
        permissions = {}
        for perm_type in PermissionType:
            permission = await self.get_permission(user_id, perm_type)
            permissions[perm_type.value] = {
                "granted": permission.granted if permission else False,
                "granted_at": permission.granted_at.isoformat() if permission and permission.granted_at else None,
                "expires_at": permission.expires_at.isoformat() if permission and permission.expires_at else None,
                "revoked": permission.revoked if permission else False
            }
        
        consents = {}
        for data_cat in DataCategory:
            status = await self.get_consent_status(user_id, data_cat)
            consents[data_cat.value] = status.value
        
        # Get data storage summary
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT data_category, COUNT(*) as count
                FROM encrypted_data 
                WHERE user_id = ?
                GROUP BY data_category
            """, (user_id,))
            data_storage = dict(cursor.fetchall())
        
        return {
            "permissions": permissions,
            "consents": consents,
            "data_storage": data_storage,
            "last_updated": datetime.now().isoformat()
        }
    
    async def _grant_permission(self, user_id: str, permission_type: PermissionType,
                              scope: Optional[Dict[str, Any]], expires_at: Optional[datetime]) -> None:
        """Grant permission to user"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_permissions 
                (user_id, permission_type, granted, granted_at, expires_at, scope)
                VALUES (?, ?, TRUE, ?, ?, ?)
            """, (user_id, permission_type.value, datetime.now(), expires_at,
                  json.dumps(scope) if scope else None))
            conn.commit()
        
        # Update cache
        if user_id not in self._permission_cache:
            self._permission_cache[user_id] = {}
        self._permission_cache[user_id][permission_type] = True
        
        await self._log_audit_event(user_id, "permission_granted", f"{permission_type.value}")
    
    async def _deny_permission(self, user_id: str, permission_type: PermissionType) -> None:
        """Deny permission request"""
        await self._log_audit_event(user_id, "permission_denied", f"{permission_type.value}")
    
    async def _expire_permission(self, user_id: str, permission_type: PermissionType) -> None:
        """Mark permission as expired"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE user_permissions 
                SET revoked = TRUE, revoked_at = ?
                WHERE user_id = ? AND permission_type = ?
            """, (datetime.now(), user_id, permission_type.value))
            conn.commit()
        
        # Update cache
        if user_id in self._permission_cache and permission_type in self._permission_cache[user_id]:
            self._permission_cache[user_id][permission_type] = False
    
    async def _record_consent(self, user_id: str, data_category: DataCategory,
                            status: ConsentStatus, details: Dict[str, Any],
                            expires_at: Optional[datetime]) -> None:
        """Record user consent"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_consent 
                (user_id, data_category, consent_status, granted_at, expires_at, consent_details, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, data_category.value, status.value,
                  datetime.now() if status == ConsentStatus.GRANTED else None,
                  expires_at, json.dumps(details), datetime.now()))
            conn.commit()
        
        # Update cache
        if user_id not in self._consent_cache:
            self._consent_cache[user_id] = {}
        self._consent_cache[user_id][data_category] = status
    
    async def _expire_consent(self, user_id: str, data_category: DataCategory) -> None:
        """Mark consent as expired"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE user_consent 
                SET consent_status = ?, updated_at = ?
                WHERE user_id = ? AND data_category = ?
            """, (ConsentStatus.EXPIRED.value, datetime.now(), user_id, data_category.value))
            conn.commit()
        
        # Update cache
        if user_id in self._consent_cache:
            self._consent_cache[user_id][data_category] = ConsentStatus.EXPIRED
    
    async def _simulate_user_permission_response(self, permission_type: PermissionType) -> bool:
        """Simulate user response to permission request (for demo purposes)"""
        # In a real implementation, this would show a UI prompt
        # For demo, grant most permissions except sensitive ones
        sensitive_permissions = {PermissionType.SCREEN_MONITOR}
        return permission_type not in sensitive_permissions
    
    async def _simulate_user_consent_response(self, data_category: DataCategory) -> ConsentStatus:
        """Simulate user response to consent request (for demo purposes)"""
        # In a real implementation, this would show a UI prompt
        # For demo, grant consent for most categories
        sensitive_categories = {DataCategory.SCREEN_CONTENT}
        if data_category in sensitive_categories:
            return ConsentStatus.DENIED
        return ConsentStatus.GRANTED
    
    def _init_default_retention_policies(self) -> Dict[DataCategory, DataRetentionPolicy]:
        """Initialize default data retention policies"""
        return {
            DataCategory.PERSONAL_INFO: DataRetentionPolicy.PERMANENT,
            DataCategory.INTERACTION_HISTORY: DataRetentionPolicy.MEDIUM_TERM,
            DataCategory.FILE_ACCESS: DataRetentionPolicy.SHORT_TERM,
            DataCategory.SCREEN_CONTENT: DataRetentionPolicy.SESSION_ONLY,
            DataCategory.PREFERENCES: DataRetentionPolicy.PERMANENT,
            DataCategory.LEARNING_DATA: DataRetentionPolicy.LONG_TERM,
            DataCategory.SYSTEM_LOGS: DataRetentionPolicy.MEDIUM_TERM,
            DataCategory.BIOMETRIC_DATA: DataRetentionPolicy.SESSION_ONLY,
            DataCategory.LOCATION_DATA: DataRetentionPolicy.SHORT_TERM,
            DataCategory.COMMUNICATION_DATA: DataRetentionPolicy.SHORT_TERM,
        }
    
    def add_privacy_change_listener(self, listener: callable) -> None:
        """Add a listener for privacy setting changes"""
        self._privacy_change_listeners.append(listener)
    
    def remove_privacy_change_listener(self, listener: callable) -> None:
        """Remove a privacy change listener"""
        if listener in self._privacy_change_listeners:
            self._privacy_change_listeners.remove(listener)
    
    async def _notify_privacy_change(self, user_id: str, setting_key: str, old_value: Any, new_value: Any) -> None:
        """Notify listeners of privacy setting changes"""
        for listener in self._privacy_change_listeners:
            try:
                await listener(user_id, setting_key, old_value, new_value)
            except Exception as e:
                logger.error(f"Error notifying privacy change listener: {e}")
    
    async def set_privacy_setting(self, user_id: str, setting_key: str, setting_value: Any,
                                privacy_level: PrivacyLevel = PrivacyLevel.INTERNAL) -> bool:
        """Set a privacy setting with immediate application"""
        try:
            # Get old value for change notification
            old_value = await self.get_privacy_setting(user_id, setting_key)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO privacy_settings 
                    (user_id, setting_key, setting_value, privacy_level, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, setting_key, json.dumps(setting_value), 
                      privacy_level.value, datetime.now()))
                conn.commit()
            
            # Update cache
            if user_id not in self._privacy_settings_cache:
                self._privacy_settings_cache[user_id] = {}
            self._privacy_settings_cache[user_id][setting_key] = setting_value
            
            # Log the change
            await self._log_audit_event(user_id, "privacy_setting_changed", setting_key,
                                       {"old_value": old_value, "new_value": setting_value})
            
            # Notify listeners for immediate application
            await self._notify_privacy_change(user_id, setting_key, old_value, setting_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set privacy setting for user {user_id}: {e}")
            return False
    
    async def get_privacy_setting(self, user_id: str, setting_key: str, default_value: Any = None) -> Any:
        """Get a privacy setting value"""
        # Check cache first
        if (user_id in self._privacy_settings_cache and 
            setting_key in self._privacy_settings_cache[user_id]):
            return self._privacy_settings_cache[user_id][setting_key]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT setting_value FROM privacy_settings 
                WHERE user_id = ? AND setting_key = ?
            """, (user_id, setting_key))
            row = cursor.fetchone()
            
            if row:
                value = json.loads(row[0])
                # Update cache
                if user_id not in self._privacy_settings_cache:
                    self._privacy_settings_cache[user_id] = {}
                self._privacy_settings_cache[user_id][setting_key] = value
                return value
            
            return default_value
    
    async def get_all_privacy_settings(self, user_id: str) -> Dict[str, Any]:
        """Get all privacy settings for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT setting_key, setting_value, privacy_level, updated_at
                FROM privacy_settings 
                WHERE user_id = ?
            """, (user_id,))
            
            settings = {}
            for row in cursor.fetchall():
                settings[row[0]] = {
                    "value": json.loads(row[1]),
                    "privacy_level": row[2],
                    "updated_at": row[3]
                }
            
            return settings
    
    async def set_data_retention_policy(self, user_id: str, data_category: DataCategory,
                                      retention_policy: DataRetentionPolicy,
                                      custom_days: Optional[int] = None,
                                      auto_delete: bool = False) -> bool:
        """Set data retention policy for a specific data category"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO data_retention_settings 
                    (user_id, data_category, retention_policy, custom_days, auto_delete, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, data_category.value, retention_policy.value,
                      custom_days, auto_delete, datetime.now()))
                conn.commit()
            
            # Update internal policy
            self._data_retention_policies[data_category] = retention_policy
            
            await self._log_audit_event(user_id, "retention_policy_updated", 
                                       data_category.value,
                                       {"policy": retention_policy.value, "custom_days": custom_days})
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set retention policy for user {user_id}: {e}")
            return False
    
    async def get_data_retention_policy(self, user_id: str, data_category: DataCategory) -> Dict[str, Any]:
        """Get data retention policy for a specific data category"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT retention_policy, custom_days, auto_delete, updated_at
                FROM data_retention_settings 
                WHERE user_id = ? AND data_category = ?
            """, (user_id, data_category.value))
            row = cursor.fetchone()
            
            if row:
                return {
                    "policy": DataRetentionPolicy(row[0]),
                    "custom_days": row[1],
                    "auto_delete": bool(row[2]),
                    "updated_at": row[3]
                }
            
            # Return default policy
            return {
                "policy": self._data_retention_policies.get(data_category, DataRetentionPolicy.MEDIUM_TERM),
                "custom_days": None,
                "auto_delete": False,
                "updated_at": None
            }
    
    async def log_data_access(self, user_id: str, data_category: DataCategory,
                            access_type: str, data_key: Optional[str] = None,
                            purpose: Optional[str] = None) -> None:
        """Log data access for transparency"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO data_access_log 
                (user_id, data_category, access_type, data_key, purpose)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, data_category.value, access_type, data_key, purpose))
            conn.commit()
    
    async def get_data_access_history(self, user_id: str, 
                                    data_category: Optional[DataCategory] = None,
                                    days: int = 30) -> List[Dict[str, Any]]:
        """Get data access history for transparency"""
        since_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            if data_category:
                cursor = conn.execute("""
                    SELECT data_category, access_type, data_key, purpose, timestamp
                    FROM data_access_log 
                    WHERE user_id = ? AND data_category = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (user_id, data_category.value, since_date))
            else:
                cursor = conn.execute("""
                    SELECT data_category, access_type, data_key, purpose, timestamp
                    FROM data_access_log 
                    WHERE user_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (user_id, since_date))
            
            return [
                {
                    "data_category": row[0],
                    "access_type": row[1],
                    "data_key": row[2],
                    "purpose": row[3],
                    "timestamp": row[4]
                }
                for row in cursor.fetchall()
            ]
    
    async def report_privacy_violation(self, user_id: str, violation_type: str,
                                     description: str, severity: str = "medium") -> str:
        """Report a privacy violation"""
        violation_id = secrets.token_urlsafe(16)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO privacy_violations 
                (user_id, violation_type, description, severity)
                VALUES (?, ?, ?, ?)
            """, (user_id, violation_type, description, severity))
            
            # Get the auto-generated ID
            cursor = conn.execute("SELECT last_insert_rowid()")
            violation_id = str(cursor.fetchone()[0])
            conn.commit()
        
        await self._log_audit_event(user_id, "privacy_violation_reported", violation_id,
                                   {"type": violation_type, "severity": severity})
        
        return violation_id
    
    async def resolve_privacy_violation(self, violation_id: str, resolution_details: str) -> bool:
        """Resolve a privacy violation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE privacy_violations 
                SET resolved = TRUE, resolution_details = ?
                WHERE id = ?
            """, (resolution_details, violation_id))
            
            if cursor.rowcount > 0:
                conn.commit()
                return True
        
        return False
    
    async def get_privacy_violations(self, user_id: str, resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get privacy violations for a user"""
        with sqlite3.connect(self.db_path) as conn:
            if resolved is not None:
                cursor = conn.execute("""
                    SELECT id, violation_type, description, severity, resolved, timestamp, resolution_details
                    FROM privacy_violations 
                    WHERE user_id = ? AND resolved = ?
                    ORDER BY timestamp DESC
                """, (user_id, resolved))
            else:
                cursor = conn.execute("""
                    SELECT id, violation_type, description, severity, resolved, timestamp, resolution_details
                    FROM privacy_violations 
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                """, (user_id,))
            
            return [
                {
                    "id": row[0],
                    "violation_type": row[1],
                    "description": row[2],
                    "severity": row[3],
                    "resolved": bool(row[4]),
                    "timestamp": row[5],
                    "resolution_details": row[6]
                }
                for row in cursor.fetchall()
            ]
    
    async def get_enhanced_privacy_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive privacy dashboard data"""
        # Get basic dashboard data
        basic_data = await self.get_privacy_dashboard_data(user_id)
        
        # Get privacy settings
        privacy_settings = await self.get_all_privacy_settings(user_id)
        
        # Get data retention policies
        retention_policies = {}
        for category in DataCategory:
            policy = await self.get_data_retention_policy(user_id, category)
            retention_policies[category.value] = policy
        
        # Get recent data access history
        access_history = await self.get_data_access_history(user_id, days=7)
        
        # Get privacy violations
        violations = await self.get_privacy_violations(user_id, resolved=False)
        
        # Calculate privacy score
        privacy_score = await self._calculate_privacy_score(user_id)
        
        return {
            **basic_data,
            "privacy_settings": privacy_settings,
            "retention_policies": retention_policies,
            "recent_access_history": access_history[:20],  # Last 20 accesses
            "active_violations": violations,
            "privacy_score": privacy_score,
            "transparency_report": {
                "total_data_accesses_7_days": len(access_history),
                "data_categories_accessed": len(set(access["data_category"] for access in access_history)),
                "privacy_violations_count": len(violations)
            }
        }
    
    async def _calculate_privacy_score(self, user_id: str) -> Dict[str, Any]:
        """Calculate a privacy score based on user's privacy settings and data usage"""
        score = 100  # Start with perfect score
        factors = []
        
        # Check permission granularity
        permissions = {}
        for perm_type in PermissionType:
            permission = await self.get_permission(user_id, perm_type)
            permissions[perm_type] = permission
        
        granted_permissions = sum(1 for p in permissions.values() if p and p.granted and not p.revoked)
        total_permissions = len(PermissionType)
        
        if granted_permissions > total_permissions * 0.8:
            score -= 10
            factors.append("Many permissions granted")
        
        # Check data retention policies
        short_retention_count = 0
        for category in DataCategory:
            policy = await self.get_data_retention_policy(user_id, category)
            if policy["policy"] in [DataRetentionPolicy.SESSION_ONLY, DataRetentionPolicy.SHORT_TERM]:
                short_retention_count += 1
        
        if short_retention_count < len(DataCategory) * 0.3:
            score -= 15
            factors.append("Long data retention periods")
        
        # Check for privacy violations
        violations = await self.get_privacy_violations(user_id, resolved=False)
        if violations:
            score -= len(violations) * 5
            factors.append(f"{len(violations)} unresolved privacy violations")
        
        # Check encryption usage
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM encrypted_data WHERE user_id = ?
            """, (user_id,))
            encrypted_count = cursor.fetchone()[0]
            
            if encrypted_count == 0:
                score -= 20
                factors.append("No encrypted data storage")
        
        return {
            "score": max(0, score),
            "factors": factors,
            "recommendations": await self._get_privacy_recommendations(user_id, score)
        }
    
    async def _get_privacy_recommendations(self, user_id: str, current_score: int) -> List[str]:
        """Get privacy improvement recommendations"""
        recommendations = []
        
        if current_score < 70:
            recommendations.append("Review and revoke unnecessary permissions")
            recommendations.append("Set shorter data retention periods for sensitive data")
            recommendations.append("Enable automatic data deletion for temporary data")
        
        if current_score < 50:
            recommendations.append("Resolve all privacy violations")
            recommendations.append("Enable encryption for all personal data")
            recommendations.append("Review data access patterns and restrict unnecessary access")
        
        # Check for specific improvements
        violations = await self.get_privacy_violations(user_id, resolved=False)
        if violations:
            recommendations.append("Address unresolved privacy violations")
        
        # Check screen monitoring
        screen_permission = await self.get_permission(user_id, PermissionType.SCREEN_MONITOR)
        if screen_permission and screen_permission.granted:
            recommendations.append("Consider limiting screen monitoring to specific applications")
        
        return recommendations
    
    async def apply_privacy_settings_immediately(self, user_id: str, settings: Dict[str, Any]) -> bool:
        """Apply privacy settings changes immediately across the system"""
        try:
            for setting_key, setting_value in settings.items():
                await self.set_privacy_setting(user_id, setting_key, setting_value)
            
            # Trigger immediate application through listeners
            await self._log_audit_event(user_id, "privacy_settings_applied_immediately", 
                                       None, {"settings_count": len(settings)})
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply privacy settings immediately for user {user_id}: {e}")
            return False
    
    async def _log_audit_event(self, user_id: str, action: str, resource: Optional[str] = None,
                             details: Optional[Dict[str, Any]] = None) -> None:
        """Log audit event"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO audit_log (user_id, action, resource, details)
                VALUES (?, ?, ?, ?)
            """, (user_id, action, resource, json.dumps(details) if details else None))
            conn.commit()