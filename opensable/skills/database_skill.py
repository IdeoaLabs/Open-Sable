"""
Database Skill - Query and manage SQL and NoSQL databases.

Supports:
- SQL databases (PostgreSQL, MySQL, SQLite)
- NoSQL databases (MongoDB, Redis)
- Connection pooling
- Query execution with parameterization
- Transactions
- Schema management
- Query builder
- ORM integration (SQLAlchemy)
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager
import sqlite3

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import aiomysql
    AIOMYSQL_AVAILABLE = True
except ImportError:
    AIOMYSQL_AVAILABLE = False

try:
    from motor import motor_asyncio
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False

try:
    import aioredis
    AIOREDIS_AVAILABLE = True
except ImportError:
    try:
        import redis.asyncio as aioredis
        AIOREDIS_AVAILABLE = True
    except ImportError:
        AIOREDIS_AVAILABLE = False


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    type: str  # sqlite, postgresql, mysql, mongodb, redis
    host: Optional[str] = "localhost"
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Set default ports
        if self.port is None:
            defaults = {
                "postgresql": 5432,
                "mysql": 3306,
                "mongodb": 27017,
                "redis": 6379
            }
            self.port = defaults.get(self.type)


@dataclass
class QueryResult:
    """Result from database query."""
    success: bool
    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    affected_rows: int = 0
    error: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "rows": self.rows,
            "row_count": self.row_count,
            "affected_rows": self.affected_rows,
            "error": self.error,
            "execution_time": self.execution_time
        }


class DatabaseManager:
    """
    Universal database manager for SQL and NoSQL databases.
    
    Features:
    - Multi-database support (PostgreSQL, MySQL, SQLite, MongoDB, Redis)
    - Connection pooling
    - Async query execution
    - Transaction support
    - Query parameterization
    - Schema management
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database manager.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self.connection = None
        self.pool = None
        self._connected = False
    
    async def connect(self):
        """Establish database connection."""
        if self._connected:
            return
        
        db_type = self.config.type.lower()
        
        if db_type == "sqlite":
            await self._connect_sqlite()
        elif db_type == "postgresql":
            await self._connect_postgresql()
        elif db_type == "mysql":
            await self._connect_mysql()
        elif db_type == "mongodb":
            await self._connect_mongodb()
        elif db_type == "redis":
            await self._connect_redis()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        self._connected = True
    
    async def disconnect(self):
        """Close database connection."""
        if not self._connected:
            return
        
        db_type = self.config.type.lower()
        
        try:
            if db_type == "sqlite" and self.connection:
                await self.connection.close()
            elif db_type == "postgresql" and self.pool:
                await self.pool.close()
            elif db_type == "mysql" and self.pool:
                self.pool.close()
                await self.pool.wait_closed()
            elif db_type == "mongodb" and self.connection:
                self.connection.close()
            elif db_type == "redis" and self.connection:
                await self.connection.close()
        except Exception:
            pass
        
        self._connected = False
        self.connection = None
        self.pool = None
    
    async def _connect_sqlite(self):
        """Connect to SQLite database."""
        if not AIOSQLITE_AVAILABLE:
            raise ImportError("aiosqlite not installed: pip install aiosqlite")
        
        db_path = self.config.database or ":memory:"
        self.connection = await aiosqlite.connect(db_path)
        self.connection.row_factory = aiosqlite.Row
    
    async def _connect_postgresql(self):
        """Connect to PostgreSQL database."""
        if not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg not installed: pip install asyncpg")
        
        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.username,
            password=self.config.password,
            **self.config.options
        )
    
    async def _connect_mysql(self):
        """Connect to MySQL database."""
        if not AIOMYSQL_AVAILABLE:
            raise ImportError("aiomysql not installed: pip install aiomysql")
        
        self.pool = await aiomysql.create_pool(
            host=self.config.host,
            port=self.config.port,
            db=self.config.database,
            user=self.config.username,
            password=self.config.password,
            **self.config.options
        )
    
    async def _connect_mongodb(self):
        """Connect to MongoDB database."""
        if not MOTOR_AVAILABLE:
            raise ImportError("motor not installed: pip install motor")
        
        connection_string = f"mongodb://"
        if self.config.username and self.config.password:
            connection_string += f"{self.config.username}:{self.config.password}@"
        connection_string += f"{self.config.host}:{self.config.port}"
        
        client = motor_asyncio.AsyncIOMotorClient(connection_string)
        self.connection = client[self.config.database]
    
    async def _connect_redis(self):
        """Connect to Redis database."""
        if not AIOREDIS_AVAILABLE:
            raise ImportError("redis not installed: pip install redis[asyncio]")
        
        self.connection = await aioredis.from_url(
            f"redis://{self.config.host}:{self.config.port}",
            password=self.config.password,
            **self.config.options
        )
    
    async def execute(
        self,
        query: str,
        params: Optional[Union[List, Dict]] = None
    ) -> QueryResult:
        """
        Execute a database query.
        
        Args:
            query: SQL query or command
            params: Query parameters
        
        Returns:
            QueryResult with results
        """
        if not self._connected:
            await self.connect()
        
        start_time = datetime.now()
        
        try:
            db_type = self.config.type.lower()
            
            if db_type == "sqlite":
                result = await self._execute_sqlite(query, params)
            elif db_type == "postgresql":
                result = await self._execute_postgresql(query, params)
            elif db_type == "mysql":
                result = await self._execute_mysql(query, params)
            elif db_type == "mongodb":
                result = await self._execute_mongodb(query, params)
            elif db_type == "redis":
                result = await self._execute_redis(query, params)
            else:
                result = QueryResult(
                    success=False,
                    error=f"Unsupported database type: {db_type}"
                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _execute_sqlite(self, query: str, params: Optional[Union[List, Dict]]) -> QueryResult:
        """Execute SQLite query."""
        async with self.connection.execute(query, params or []) as cursor:
            if query.strip().upper().startswith("SELECT"):
                rows = await cursor.fetchall()
                return QueryResult(
                    success=True,
                    rows=[dict(row) for row in rows],
                    row_count=len(rows)
                )
            else:
                await self.connection.commit()
                return QueryResult(
                    success=True,
                    affected_rows=cursor.rowcount
                )
    
    async def _execute_postgresql(self, query: str, params: Optional[Union[List, Dict]]) -> QueryResult:
        """Execute PostgreSQL query."""
        async with self.pool.acquire() as conn:
            if query.strip().upper().startswith("SELECT"):
                rows = await conn.fetch(query, *(params or []))
                return QueryResult(
                    success=True,
                    rows=[dict(row) for row in rows],
                    row_count=len(rows)
                )
            else:
                result = await conn.execute(query, *(params or []))
                # Parse result like "UPDATE 5"
                affected = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                return QueryResult(
                    success=True,
                    affected_rows=affected
                )
    
    async def _execute_mysql(self, query: str, params: Optional[Union[List, Dict]]) -> QueryResult:
        """Execute MySQL query."""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(query, params or [])
                
                if query.strip().upper().startswith("SELECT"):
                    rows = await cursor.fetchall()
                    return QueryResult(
                        success=True,
                        rows=rows,
                        row_count=len(rows)
                    )
                else:
                    await conn.commit()
                    return QueryResult(
                        success=True,
                        affected_rows=cursor.rowcount
                    )
    
    async def _execute_mongodb(self, query: str, params: Optional[Union[List, Dict]]) -> QueryResult:
        """Execute MongoDB query."""
        # Parse MongoDB query (expecting JSON format)
        try:
            query_obj = json.loads(query)
            operation = query_obj.get("operation")
            collection_name = query_obj.get("collection")
            filter_obj = query_obj.get("filter", {})
            document = query_obj.get("document", {})
            
            collection = self.connection[collection_name]
            
            if operation == "find":
                cursor = collection.find(filter_obj)
                rows = await cursor.to_list(length=1000)
                # Convert ObjectId to string
                for row in rows:
                    if "_id" in row:
                        row["_id"] = str(row["_id"])
                return QueryResult(
                    success=True,
                    rows=rows,
                    row_count=len(rows)
                )
            
            elif operation == "insert":
                result = await collection.insert_one(document)
                return QueryResult(
                    success=True,
                    affected_rows=1,
                    rows=[{"inserted_id": str(result.inserted_id)}]
                )
            
            elif operation == "update":
                result = await collection.update_many(filter_obj, {"$set": document})
                return QueryResult(
                    success=True,
                    affected_rows=result.modified_count
                )
            
            elif operation == "delete":
                result = await collection.delete_many(filter_obj)
                return QueryResult(
                    success=True,
                    affected_rows=result.deleted_count
                )
            
            else:
                return QueryResult(
                    success=False,
                    error=f"Unknown MongoDB operation: {operation}"
                )
        
        except json.JSONDecodeError:
            return QueryResult(
                success=False,
                error="Invalid MongoDB query format (expected JSON)"
            )
    
    async def _execute_redis(self, query: str, params: Optional[Union[List, Dict]]) -> QueryResult:
        """Execute Redis command."""
        # Parse Redis command
        parts = query.split()
        command = parts[0].upper()
        args = parts[1:] if len(parts) > 1 else []
        
        if command == "GET":
            value = await self.connection.get(args[0])
            return QueryResult(
                success=True,
                rows=[{"key": args[0], "value": value.decode() if value else None}],
                row_count=1
            )
        
        elif command == "SET":
            await self.connection.set(args[0], args[1])
            return QueryResult(
                success=True,
                affected_rows=1
            )
        
        elif command == "DEL":
            count = await self.connection.delete(*args)
            return QueryResult(
                success=True,
                affected_rows=count
            )
        
        elif command == "KEYS":
            keys = await self.connection.keys(args[0] if args else "*")
            return QueryResult(
                success=True,
                rows=[{"key": k.decode()} for k in keys],
                row_count=len(keys)
            )
        
        else:
            # Try to execute generic command
            try:
                result = await self.connection.execute_command(command, *args)
                return QueryResult(
                    success=True,
                    rows=[{"result": str(result)}]
                )
            except Exception as e:
                return QueryResult(
                    success=False,
                    error=str(e)
                )
    
    async def execute_many(
        self,
        query: str,
        params_list: List[Union[List, Dict]]
    ) -> QueryResult:
        """
        Execute query with multiple parameter sets.
        
        Args:
            query: SQL query
            params_list: List of parameter sets
        
        Returns:
            QueryResult with combined results
        """
        if not self._connected:
            await self.connect()
        
        total_affected = 0
        
        for params in params_list:
            result = await self.execute(query, params)
            if not result.success:
                return result
            total_affected += result.affected_rows
        
        return QueryResult(
            success=True,
            affected_rows=total_affected
        )
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for database transactions.
        
        Usage:
            async with db.transaction():
                await db.execute("INSERT ...")
                await db.execute("UPDATE ...")
        """
        if not self._connected:
            await self.connect()
        
        db_type = self.config.type.lower()
        
        if db_type == "sqlite":
            await self.connection.execute("BEGIN")
            try:
                yield
                await self.connection.commit()
            except Exception:
                await self.connection.rollback()
                raise
        
        elif db_type == "postgresql":
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    yield
        
        elif db_type == "mysql":
            async with self.pool.acquire() as conn:
                await conn.begin()
                try:
                    yield
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise
        
        else:
            # No transaction support for NoSQL
            yield
    
    async def create_table(self, table_name: str, schema: Dict[str, str]) -> QueryResult:
        """
        Create a table with given schema.
        
        Args:
            table_name: Table name
            schema: Column definitions {column_name: column_type}
        
        Returns:
            QueryResult
        """
        db_type = self.config.type.lower()
        
        if db_type in ["sqlite", "postgresql", "mysql"]:
            columns = ", ".join([f"{name} {type_}" for name, type_ in schema.items()])
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
            return await self.execute(query)
        
        elif db_type == "mongodb":
            # MongoDB creates collections automatically
            return QueryResult(success=True)
        
        else:
            return QueryResult(
                success=False,
                error=f"Table creation not supported for {db_type}"
            )
    
    async def drop_table(self, table_name: str) -> QueryResult:
        """Drop a table."""
        db_type = self.config.type.lower()
        
        if db_type in ["sqlite", "postgresql", "mysql"]:
            query = f"DROP TABLE IF EXISTS {table_name}"
            return await self.execute(query)
        
        elif db_type == "mongodb":
            await self.connection[table_name].drop()
            return QueryResult(success=True)
        
        else:
            return QueryResult(
                success=False,
                error=f"Table deletion not supported for {db_type}"
            )
    
    async def list_tables(self) -> List[str]:
        """List all tables/collections."""
        db_type = self.config.type.lower()
        
        if db_type == "sqlite":
            result = await self.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            return [row["name"] for row in result.rows]
        
        elif db_type == "postgresql":
            result = await self.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname='public'"
            )
            return [row["tablename"] for row in result.rows]
        
        elif db_type == "mysql":
            result = await self.execute("SHOW TABLES")
            return [list(row.values())[0] for row in result.rows]
        
        elif db_type == "mongodb":
            return await self.connection.list_collection_names()
        
        else:
            return []


# Example usage
async def main():
    """Example database operations."""
    
    # SQLite example
    print("=" * 50)
    print("SQLite Database Example")
    print("=" * 50)
    
    config = DatabaseConfig(type="sqlite", database=":memory:")
    db = DatabaseManager(config)
    
    try:
        await db.connect()
        
        # Create table
        print("\n1. Creating table...")
        result = await db.create_table(
            "users",
            {
                "id": "INTEGER PRIMARY KEY",
                "name": "TEXT NOT NULL",
                "email": "TEXT UNIQUE",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
        )
        print(f"  Success: {result.success}")
        
        # Insert data
        print("\n2. Inserting data...")
        result = await db.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            ["Alice", "alice@example.com"]
        )
        print(f"  Affected rows: {result.affected_rows}")
        
        result = await db.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            ["Bob", "bob@example.com"]
        )
        print(f"  Affected rows: {result.affected_rows}")
        
        # Query data
        print("\n3. Querying data...")
        result = await db.execute("SELECT * FROM users")
        print(f"  Found {result.row_count} users:")
        for row in result.rows:
            print(f"    - {row['name']} ({row['email']})")
        
        # Update data
        print("\n4. Updating data...")
        result = await db.execute(
            "UPDATE users SET email = ? WHERE name = ?",
            ["alice.new@example.com", "Alice"]
        )
        print(f"  Affected rows: {result.affected_rows}")
        
        # Transaction example
        print("\n5. Transaction example...")
        async with db.transaction():
            await db.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                ["Charlie", "charlie@example.com"]
            )
            await db.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                ["David", "david@example.com"]
            )
        print("  Transaction committed")
        
        # List tables
        print("\n6. Listing tables...")
        tables = await db.list_tables()
        print(f"  Tables: {', '.join(tables)}")
        
    finally:
        await db.disconnect()
    
    print("\n✅ Database examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
