"""
Простое хранилище чатов в памяти.
В будущем можно заменить на БД.
"""
from typing import Dict, List, Optional
from datetime import datetime
import uuid


class ChatMessage:
    def __init__(
        self, 
        role: str, 
        content: str, 
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}  # sources, tool_calls, etc.
    
    def to_dict(self):
        result = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
        # Добавляем metadata если есть и он не пустой
        if self.metadata and len(self.metadata) > 0:
            result["metadata"] = self.metadata
        return result


class Chat:
    def __init__(self, chat_id: str, title: str = "Новый чат"):
        self.id = chat_id
        self.title = title
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        message = ChatMessage(role, content, metadata=metadata)
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def to_dict(self, include_messages: bool = False):
        result = {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": len(self.messages)
        }
        if include_messages:
            result["messages"] = [msg.to_dict() for msg in self.messages]
        return result


class ChatStorage:
    """Singleton хранилище чатов в памяти"""
    
    _instance = None
    _chats: Dict[str, Chat] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatStorage, cls).__new__(cls)
            cls._instance._chats = {}
        return cls._instance
    
    def create_chat(self, title: str = "Новый чат") -> Chat:
        """Создает новый чат"""
        chat_id = str(uuid.uuid4())
        chat = Chat(chat_id, title)
        self._chats[chat_id] = chat
        print(f"📝 Создан новый чат: {chat_id} - {title}")
        return chat
    
    def get_chat(self, chat_id: str) -> Optional[Chat]:
        """Получает чат по ID"""
        return self._chats.get(chat_id)
    
    def get_all_chats(self) -> List[Chat]:
        """Получает все чаты, отсортированные по дате обновления"""
        chats = list(self._chats.values())
        chats.sort(key=lambda x: x.updated_at, reverse=True)
        return chats
    
    def update_chat_title(self, chat_id: str, title: str) -> bool:
        """Обновляет название чата"""
        chat = self.get_chat(chat_id)
        if chat:
            chat.title = title
            chat.updated_at = datetime.now()
            print(f"✏️ Обновлено название чата {chat_id}: {title}")
            return True
        return False
    
    def delete_chat(self, chat_id: str) -> bool:
        """Удаляет чат"""
        if chat_id in self._chats:
            del self._chats[chat_id]
            print(f"🗑️ Удален чат: {chat_id}")
            return True
        return False
    
    def add_message(self, chat_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Добавляет сообщение в чат"""
        chat = self.get_chat(chat_id)
        if chat:
            if metadata:
                print(f"[STORAGE] Adding {role} message with metadata keys: {list(metadata.keys())}")
            else:
                print(f"[STORAGE] Adding {role} message without metadata")
            chat.add_message(role, content, metadata=metadata)
            return True
        return False
    
    def get_messages(self, chat_id: str) -> List[Dict]:
        """Получает все сообщения чата"""
        chat = self.get_chat(chat_id)
        if chat:
            return [msg.to_dict() for msg in chat.messages]
        return []


# Singleton instance
chat_storage = ChatStorage()

