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


class PrivacySecurityManager:
    """Manages privacy controls, security, and user permissions"""
    
    def __init__(self, db_path: str = "personal_assistant.db", encryption_key: Optional[bytes] = None):
        self.db_path = db_path
        self._init_database()
        self._init_encryption(encryption_key)
        self._permission_cache: Dict[str, Dict[PermissionType, bool]] = {}
        self._consent_cache: Dict[str, Dict[DataCategory, ConsentStatus]] = {}
    
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
                                  data: Dict[str, Any], data_category: DataCategory) -> bool:
        """Encrypt and store personal data"""
        try:
            # Serialize and encrypt data
            data_json = json.dumps(data)
            encrypted_data = self.cipher_suite.encrypt(data_json.encode())
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO encrypted_data 
                    (user_id, data_key, encrypted_content, data_category, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, data_key, encrypted_data, data_category.value, datetime.now()))
                conn.commit()
            
            await self._log_audit_event(user_id, "data_encrypted", data_key)
            return True
            
        except Exception as e:
            logger.error(f"Failed to encrypt data for user {user_id}: {e}")
            return False
    
    async def decrypt_personal_data(self, user_id: str, data_key: str) -> Optional[Dict[str, Any]]:
        """Decrypt and retrieve personal data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT encrypted_content FROM encrypted_data 
                    WHERE user_id = ? AND data_key = ?
                """, (user_id, data_key))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Decrypt data
                decrypted_data = self.cipher_suite.decrypt(row[0])
                data = json.loads(decrypted_data.decode())
                
                await self._log_audit_event(user_id, "data_decrypted", data_key)
                return data
                
        except Exception as e:
            logger.error(f"Failed to decrypt data for user {user_id}: {e}")
            return None
    
    async def request_data_deletion(self, user_id: str, data_categories: List[DataCategory],
                                  reason: Optional[str] = None) -> str:
        """Request deletion of user data"""
        request_id = secrets.token_urlsafe(16)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO data_deletion_requests 
                (id, user_id, data_categories, details)
                VALUES (?, ?, ?, ?)
            """, (request_id, user_id, 
                  json.dumps([cat.value for cat in data_categories]),
                  json.dumps({"reason": reason})))
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
        """Delete user data by category"""
        with sqlite3.connect(self.db_path) as conn:
            if category == DataCategory.INTERACTION_HISTORY:
                conn.execute("DELETE FROM user_interactions WHERE user_id = ?", (user_id,))
            elif category == DataCategory.PERSONAL_INFO:
                conn.execute("DELETE FROM user_contexts WHERE user_id = ?", (user_id,))
            elif category == DataCategory.FILE_ACCESS:
                conn.execute("DELETE FROM encrypted_data WHERE user_id = ? AND data_category = ?", 
                           (user_id, DataCategory.FILE_ACCESS.value))
            elif category == DataCategory.SCREEN_CONTENT:
                conn.execute("DELETE FROM encrypted_data WHERE user_id = ? AND data_category = ?", 
                           (user_id, DataCategory.SCREEN_CONTENT.value))
            elif category == DataCategory.PREFERENCES:
                # Update context to remove preferences
                pass
            elif category == DataCategory.LEARNING_DATA:
                conn.execute("DELETE FROM encrypted_data WHERE user_id = ? AND data_category = ?", 
                           (user_id, DataCategory.LEARNING_DATA.value))
            
            conn.commit()
    
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
    
    async def _log_audit_event(self, user_id: str, action: str, resource: Optional[str] = None,
                             details: Optional[Dict[str, Any]] = None) -> None:
        """Log audit event"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO audit_log (user_id, action, resource, details)
                VALUES (?, ?, ?, ?)
            """, (user_id, action, resource, json.dumps(details) if details else None))
            conn.commit()