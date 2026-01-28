"""
MySQL 事件存储层。

提供事件的持久化存储、分页查询和清理功能。
"""
from __future__ import annotations

import mysql.connector
from typing import TYPE_CHECKING, Optional
from contextlib import contextmanager
from datetime import datetime, timezone

from src.run.log import get_logger

if TYPE_CHECKING:
    from src.classes.event import Event


def _format_time(ts: float) -> str:
    """将 timestamp float 转换为 MySQL 兼容的 UTC 字符串"""
    return datetime.fromtimestamp(ts, timezone.utc).strftime(
        '%Y-%m-%d %H:%M:%S.%f'
    )


def _parse_time(ts_str: str) -> float:
    """将 MySQL 时间字符串解析为 timestamp float"""
    if not ts_str:
        return 0.0
    try:
        dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            # 尝试不带微秒的格式
            dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return 0.0
    # 假设数据库存的是 UTC (naive time string from mysql usually treated as such)
    return dt.replace(tzinfo=timezone.utc).timestamp()


class EventStorageMySQL:
    """
    MySQL 事件存储层。

    提供：
    - 实时写入事件
    - 分页查询（cursor-based）
    - 按角色/角色对查询
    - 历史清理
    """

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        """
        初始化数据库连接，创建表（如不存在）。

        Args:
            host: MySQL 主机地址。
            port: MySQL 端口。
            user: MySQL 用户名。
            password: MySQL 密码。
            database: MySQL 数据库名。
        """
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._conn: Optional[mysql.connector.connection.MySQLConnection] = None
        self._logger = get_logger().logger
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库连接和表结构。"""
        try:
            # 连接数据库
            self._conn = mysql.connector.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                database=self._database
            )

            # 创建表
            self._create_tables()
            self._logger.info(f"EventStorageMySQL initialized: {self._host}:{self._port}/{self._database}")
        except Exception as e:
            self._logger.error(f"Failed to initialize EventStorageMySQL: {e}")
            raise

    def _create_tables(self) -> None:
        """创建必要的表结构。"""
        if not self._conn:
            raise ValueError("Database connection not initialized")

        cursor = self._conn.cursor()
        try:
            # 创建事件表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id VARCHAR(255) PRIMARY KEY,
                    month_stamp BIGINT NOT NULL,
                    content TEXT NOT NULL,
                    is_major BOOLEAN DEFAULT FALSE,
                    is_story BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)

            # 创建事件-角色关联表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_avatars (
                    event_id VARCHAR(255) NOT NULL,
                    avatar_id VARCHAR(255) NOT NULL,
                    PRIMARY KEY (event_id, avatar_id),
                    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)

            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_month_stamp ON events(month_stamp DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_is_major ON events(is_major);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_avatars_avatar_id ON event_avatars(avatar_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_avatars_event_id ON event_avatars(event_id);")

            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            self._logger.error(f"Failed to create tables: {e}")
            raise
        finally:
            cursor.close()

    @contextmanager
    def _transaction(self):
        """事务上下文管理器。"""
        if not self._conn:
            raise ValueError("Database connection not initialized")

        cursor = self._conn.cursor(dictionary=True)
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def add_event(self, event: "Event") -> bool:
        """
        写入单个事件。

        失败时记录日志并返回 False，不抛异常。

        Args:
            event: 要写入的事件对象。

        Returns:
            写入是否成功。
        """
        if not self._conn:
            self._logger.error("EventStorageMySQL not initialized")
            return False

        try:
            with self._transaction() as cursor:
                # 插入事件主表
                cursor.execute(
                    """
                    INSERT IGNORE INTO events (id, month_stamp, content, is_major, is_story, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event.id,
                        int(event.month_stamp),
                        event.content,
                        event.is_major,
                        event.is_story,
                        _format_time(event.created_at),
                    )
                )

                # 插入关联表
                if event.related_avatars:
                    for avatar_id in event.related_avatars:
                        cursor.execute(
                            """
                            INSERT IGNORE INTO event_avatars (event_id, avatar_id)
                            VALUES (%s, %s)
                            """,
                            (event.id, str(avatar_id))
                        )
            return True
        except Exception as e:
            self._logger.error(f"Failed to write event {event.id}: {e}")
            return False

    def _parse_cursor(self, cursor: str) -> tuple[int, str]:
        """
        解析复合 cursor。

        格式: {month_stamp}_{event_id}

        Returns:
            (month_stamp, event_id)
        """
        parts = cursor.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid cursor format: {cursor}")
        return int(parts[0]), parts[1]

    def _make_cursor(self, month_stamp: int, event_id: str) -> str:
        """生成复合 cursor。"""
        return f"{month_stamp}_{event_id}"

    def get_events(
        self,
        avatar_id: Optional[str] = None,
        avatar_id_pair: Optional[tuple[str, str]] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> tuple[list["Event"], Optional[str]]:
        """
        分页查询事件。

        Args:
            avatar_id: 按单个角色筛选。
            avatar_id_pair: Pair 查询（两个角色之间的事件）。
            cursor: 分页 cursor，获取该位置之前的事件。
            limit: 每页数量。

        Returns:
            (events, next_cursor)，next_cursor 为 None 表示没有更多。
        """
        from src.classes.event import Event
        from src.classes.calendar import MonthStamp

        if not self._conn:
            return [], None

        try:
            cursor_obj = self._conn.cursor(dictionary=True)
            try:
                # 构建查询
                params: list = []

                if avatar_id_pair:
                    # Pair 查询：两个角色都相关的事件
                    id1, id2 = avatar_id_pair
                    base_query = """
                        SELECT DISTINCT e.id, e.month_stamp, e.content, e.is_major, e.is_story, e.created_at
                        FROM events e
                        JOIN event_avatars ea1 ON e.id = ea1.event_id AND ea1.avatar_id = %s
                        JOIN event_avatars ea2 ON e.id = ea2.event_id AND ea2.avatar_id = %s
                    """
                    params.extend([id1, id2])
                elif avatar_id:
                    # 单角色查询
                    base_query = """
                        SELECT DISTINCT e.id, e.month_stamp, e.content, e.is_major, e.is_story, e.created_at
                        FROM events e
                        JOIN event_avatars ea ON e.id = ea.event_id AND ea.avatar_id = %s
                    """
                    params.append(avatar_id)
                else:
                    # 全部事件
                    base_query = """
                        SELECT id, month_stamp, content, is_major, is_story, created_at
                        FROM events e
                    """

                # Cursor 条件（获取更旧的事件）
                where_clauses = []
                if cursor:
                    cursor_month, cursor_event_id = self._parse_cursor(cursor)
                    where_clauses.append(
                        "(e.month_stamp < %s OR (e.month_stamp = %s AND e.id < %s))"
                    )
                    params.extend([cursor_month, cursor_month, cursor_event_id])

                # 组装 WHERE
                if where_clauses:
                    base_query += " WHERE " + " AND ".join(where_clauses)

                # 排序和分页（最新的在前，向上加载更旧的）
                base_query += " ORDER BY e.month_stamp DESC, e.id DESC LIMIT %s"
                params.append(limit + 1)  # 多取一条判断是否有更多

                cursor_obj.execute(base_query, params)
                rows = cursor_obj.fetchall()

                # 判断是否有更多
                has_more = len(rows) > limit
                if has_more:
                    rows = rows[:limit]

                # 构建事件对象
                events = []
                last_event_id = None
                last_month_stamp = None
                for row in rows:
                    # 获取关联的 avatar IDs
                    avatar_cursor = self._conn.cursor(dictionary=True)
                    try:
                        avatar_cursor.execute(
                            "SELECT avatar_id FROM event_avatars WHERE event_id = %s",
                            (row["id"],)
                        )
                        avatar_rows = avatar_cursor.fetchall()
                        related_avatars = [r["avatar_id"] for r in avatar_rows]
                    finally:
                        avatar_cursor.close()

                    event = Event(
                        month_stamp=MonthStamp(row["month_stamp"]),
                        content=row["content"],
                        related_avatars=related_avatars if related_avatars else None,
                        is_major=bool(row["is_major"]),
                        is_story=bool(row["is_story"]),
                        id=row["id"],
                        created_at=_parse_time(row["created_at"]),
                    )
                    events.append(event)
                    last_event_id = row["id"]
                    last_month_stamp = row["month_stamp"]

                # 生成 next_cursor
                next_cursor = None
                if has_more and last_event_id is not None:
                    next_cursor = self._make_cursor(last_month_stamp, last_event_id)

                return events, next_cursor
            finally:
                cursor_obj.close()
        except Exception as e:
            self._logger.error(f"Failed to query events: {e}")
            return [], None

    def get_events_by_avatar(self, avatar_id: str, limit: int = 50) -> list["Event"]:
        """
        后端用：获取角色相关事件（供 LLM prompt 使用）。

        返回最新的 N 条，按时间正序排列。
        """
        events, _ = self.get_events(avatar_id=avatar_id, limit=limit)
        return list(reversed(events))  # 转为时间正序

    def get_events_between(self, id1: str, id2: str, limit: int = 50) -> list["Event"]:
        """
        后端用：获取两角色之间的事件。

        返回最新的 N 条，按时间正序排列。
        """
        events, _ = self.get_events(avatar_id_pair=(id1, id2), limit=limit)
        return list(reversed(events))  # 转为时间正序

    def get_major_events_by_avatar(self, avatar_id: str, limit: int = 10) -> list["Event"]:
        """获取角色的大事（长期记忆）。"""
        from src.classes.event import Event
        from src.classes.calendar import MonthStamp

        if not self._conn:
            return []

        try:
            cursor = self._conn.cursor(dictionary=True)
            try:
                cursor.execute(
                    """
                    SELECT DISTINCT e.id, e.month_stamp, e.content, e.is_major, e.is_story, e.created_at
                    FROM events e
                    JOIN event_avatars ea ON e.id = ea.event_id AND ea.avatar_id = %s
                    WHERE e.is_major = TRUE AND e.is_story = FALSE
                    ORDER BY e.month_stamp DESC
                    LIMIT %s
                    """,
                    (avatar_id, limit)
                )
                rows = cursor.fetchall()

                events = []
                for row in rows:
                    avatar_cursor = self._conn.cursor(dictionary=True)
                    try:
                        avatar_cursor.execute(
                            "SELECT avatar_id FROM event_avatars WHERE event_id = %s",
                            (row["id"],)
                        )
                        avatar_rows = avatar_cursor.fetchall()
                        related_avatars = [r["avatar_id"] for r in avatar_rows]
                    finally:
                        avatar_cursor.close()

                    event = Event(
                        month_stamp=MonthStamp(row["month_stamp"]),
                        content=row["content"],
                        related_avatars=related_avatars if related_avatars else None,
                        is_major=bool(row["is_major"]),
                        is_story=bool(row["is_story"]),
                        id=row["id"],
                        created_at=_parse_time(row["created_at"]),
                    )
                    events.append(event)

                return list(reversed(events))  # 时间正序
            finally:
                cursor.close()
        except Exception as e:
            self._logger.error(f"Failed to query major events: {e}")
            return []

    def get_minor_events_by_avatar(self, avatar_id: str, limit: int = 10) -> list["Event"]:
        """获取角色的小事（短期记忆，包括故事）。"""
        from src.classes.event import Event
        from src.classes.calendar import MonthStamp

        if not self._conn:
            return []

        try:
            cursor = self._conn.cursor(dictionary=True)
            try:
                cursor.execute(
                    """
                    SELECT DISTINCT e.id, e.month_stamp, e.content, e.is_major, e.is_story, e.created_at
                    FROM events e
                    JOIN event_avatars ea ON e.id = ea.event_id AND ea.avatar_id = %s
                    WHERE e.is_major = FALSE OR e.is_story = TRUE
                    ORDER BY e.month_stamp DESC
                    LIMIT %s
                    """,
                    (avatar_id, limit)
                )
                rows = cursor.fetchall()

                events = []
                for row in rows:
                    avatar_cursor = self._conn.cursor(dictionary=True)
                    try:
                        avatar_cursor.execute(
                            "SELECT avatar_id FROM event_avatars WHERE event_id = %s",
                            (row["id"],)
                        )
                        avatar_rows = avatar_cursor.fetchall()
                        related_avatars = [r["avatar_id"] for r in avatar_rows]
                    finally:
                        avatar_cursor.close()

                    event = Event(
                        month_stamp=MonthStamp(row["month_stamp"]),
                        content=row["content"],
                        related_avatars=related_avatars if related_avatars else None,
                        is_major=bool(row["is_major"]),
                        is_story=bool(row["is_story"]),
                        id=row["id"],
                        created_at=_parse_time(row["created_at"]),
                    )
                    events.append(event)

                return list(reversed(events))  # 时间正序
            finally:
                cursor.close()
        except Exception as e:
            self._logger.error(f"Failed to query minor events: {e}")
            return []

    def get_major_events_between(self, id1: str, id2: str, limit: int = 10) -> list["Event"]:
        """获取两个角色之间的大事（长期记忆）。"""
        from src.classes.event import Event
        from src.classes.calendar import MonthStamp

        if not self._conn:
            return []

        try:
            cursor = self._conn.cursor(dictionary=True)
            try:
                cursor.execute(
                    """
                    SELECT DISTINCT e.id, e.month_stamp, e.content, e.is_major, e.is_story, e.created_at
                    FROM events e
                    JOIN event_avatars ea1 ON e.id = ea1.event_id AND ea1.avatar_id = %s
                    JOIN event_avatars ea2 ON e.id = ea2.event_id AND ea2.avatar_id = %s
                    WHERE e.is_major = TRUE AND e.is_story = FALSE
                    ORDER BY e.month_stamp DESC
                    LIMIT %s
                    """,
                    (id1, id2, limit)
                )
                rows = cursor.fetchall()

                events = []
                for row in rows:
                    avatar_cursor = self._conn.cursor(dictionary=True)
                    try:
                        avatar_cursor.execute(
                            "SELECT avatar_id FROM event_avatars WHERE event_id = %s",
                            (row["id"],)
                        )
                        avatar_rows = avatar_cursor.fetchall()
                        related_avatars = [r["avatar_id"] for r in avatar_rows]
                    finally:
                        avatar_cursor.close()

                    event = Event(
                        month_stamp=MonthStamp(row["month_stamp"]),
                        content=row["content"],
                        related_avatars=related_avatars if related_avatars else None,
                        is_major=bool(row["is_major"]),
                        is_story=bool(row["is_story"]),
                        id=row["id"],
                        created_at=_parse_time(row["created_at"]),
                    )
                    events.append(event)

                return list(reversed(events))  # 时间正序
            finally:
                cursor.close()
        except Exception as e:
            self._logger.error(f"Failed to query major events between: {e}")
            return []

    def get_minor_events_between(self, id1: str, id2: str, limit: int = 10) -> list["Event"]:
        """获取两个角色之间的小事（短期记忆）。"""
        from src.classes.event import Event
        from src.classes.calendar import MonthStamp

        if not self._conn:
            return []

        try:
            cursor = self._conn.cursor(dictionary=True)
            try:
                cursor.execute(
                    """
                    SELECT DISTINCT e.id, e.month_stamp, e.content, e.is_major, e.is_story, e.created_at
                    FROM events e
                    JOIN event_avatars ea1 ON e.id = ea1.event_id AND ea1.avatar_id = %s
                    JOIN event_avatars ea2 ON e.id = ea2.event_id AND ea2.avatar_id = %s
                    WHERE e.is_major = FALSE OR e.is_story = TRUE
                    ORDER BY e.month_stamp DESC
                    LIMIT %s
                    """,
                    (id1, id2, limit)
                )
                rows = cursor.fetchall()

                events = []
                for row in rows:
                    avatar_cursor = self._conn.cursor(dictionary=True)
                    try:
                        avatar_cursor.execute(
                            "SELECT avatar_id FROM event_avatars WHERE event_id = %s",
                            (row["id"],)
                        )
                        avatar_rows = avatar_cursor.fetchall()
                        related_avatars = [r["avatar_id"] for r in avatar_rows]
                    finally:
                        avatar_cursor.close()

                    event = Event(
                        month_stamp=MonthStamp(row["month_stamp"]),
                        content=row["content"],
                        related_avatars=related_avatars if related_avatars else None,
                        is_major=bool(row["is_major"]),
                        is_story=bool(row["is_story"]),
                        id=row["id"],
                        created_at=_parse_time(row["created_at"]),
                    )
                    events.append(event)

                return list(reversed(events))  # 时间正序
            finally:
                cursor.close()
        except Exception as e:
            self._logger.error(f"Failed to query minor events between: {e}")
            return []

    def get_recent_events(self, limit: int = 100) -> list["Event"]:
        """获取最近的事件（供初始状态 API 使用）。"""
        events, _ = self.get_events(limit=limit)
        return list(reversed(events))  # 时间正序

    def cleanup(self, keep_major: bool = True, before_month_stamp: Optional[int] = None) -> int:
        """
        清理事件。

        Args:
            keep_major: 是否保留大事。
            before_month_stamp: 删除此时间之前的事件。

        Returns:
            删除的事件数量。
        """
        if not self._conn:
            return 0

        try:
            cursor = self._conn.cursor()
            try:
                conditions = []
                params: list = []

                if keep_major:
                    conditions.append("is_major = FALSE")

                if before_month_stamp is not None:
                    conditions.append("month_stamp < %s")
                    params.append(before_month_stamp)

                # 如果没有条件且要保留大事，则无需删除任何内容
                if not conditions and keep_major:
                    return 0

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                cursor.execute(
                    f"DELETE FROM events WHERE {where_clause}",
                    params
                )
                deleted = cursor.rowcount
                self._conn.commit()

                self._logger.info(f"Cleaned up {deleted} events")
                return deleted
            finally:
                cursor.close()
        except Exception as e:
            self._logger.error(f"Failed to cleanup events: {e}")
            return 0

    def count(self) -> int:
        """获取事件总数。"""
        if not self._conn:
            return 0
        try:
            cursor = self._conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM events")
                row = cursor.fetchone()
                return row[0] if row else 0
            finally:
                cursor.close()
        except Exception:
            return 0

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            try:
                self._conn.close()
                self._logger.info("EventStorageMySQL closed")
            except Exception as e:
                self._logger.error(f"Failed to close EventStorageMySQL: {e}")
            finally:
                self._conn = None
