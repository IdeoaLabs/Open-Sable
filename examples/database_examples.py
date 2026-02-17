"""
Database Examples - SQL and NoSQL database operations.

Demonstrates SQLite, PostgreSQL, MySQL, MongoDB, and Redis operations.
"""

import asyncio
from skills.database_skill import DatabaseManager, DatabaseConfig, DatabaseType


async def main():
    """Run database examples."""
    
    print("=" * 60)
    print("Database Examples")
    print("=" * 60)
    
    # Example 1: SQLite operations
    print("\n1. SQLite Operations")
    print("-" * 40)
    
    sqlite_config = DatabaseConfig(
        type=DatabaseType.SQLITE,
        database="example.db"
    )
    
    sqlite_manager = DatabaseManager(sqlite_config)
    await sqlite_manager.connect()
    
    # Create table
    await sqlite_manager.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            age INTEGER
        )
    """)
    print("Created users table")
    
    # Insert data
    await sqlite_manager.execute(
        "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
        params=("Alice", "alice@example.com", 30)
    )
    await sqlite_manager.execute(
        "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
        params=("Bob", "bob@example.com", 25)
    )
    print("Inserted 2 users")
    
    # Query data
    result = await sqlite_manager.query("SELECT * FROM users")
    print(f"Users: {len(result.rows)}")
    for row in result.rows:
        print(f"  - {row}")
    
    # Update
    await sqlite_manager.execute(
        "UPDATE users SET age = ? WHERE name = ?",
        params=(31, "Alice")
    )
    print("Updated Alice's age")
    
    # Delete
    await sqlite_manager.execute(
        "DELETE FROM users WHERE name = ?",
        params=("Bob",)
    )
    print("Deleted Bob")
    
    await sqlite_manager.disconnect()
    
    # Example 2: Transactions
    print("\n2. Transaction Example")
    print("-" * 40)
    
    sqlite_manager = DatabaseManager(sqlite_config)
    await sqlite_manager.connect()
    
    try:
        async with sqlite_manager.transaction():
            await sqlite_manager.execute(
                "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
                params=("Charlie", "charlie@example.com", 28)
            )
            await sqlite_manager.execute(
                "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
                params=("Diana", "diana@example.com", 32)
            )
            print("Transaction committed: Added 2 users")
    except Exception as e:
        print(f"Transaction rolled back: {e}")
    
    result = await sqlite_manager.query("SELECT COUNT(*) as count FROM users")
    print(f"Total users: {result.rows[0]['count']}")
    
    await sqlite_manager.disconnect()
    
    # Example 3: Connection pooling
    print("\n3. Connection Pool")
    print("-" * 40)
    
    pool_config = DatabaseConfig(
        type=DatabaseType.SQLITE,
        database="pooled.db",
        pool_size=5
    )
    
    pool_manager = DatabaseManager(pool_config)
    await pool_manager.connect()
    
    # Simulate concurrent queries
    tasks = []
    for i in range(10):
        task = pool_manager.query("SELECT 1 as value")
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    print(f"Executed {len(results)} concurrent queries")
    
    await pool_manager.disconnect()
    
    # Example 4: MongoDB operations (simulated)
    print("\n4. MongoDB Operations")
    print("-" * 40)
    
    mongo_config = DatabaseConfig(
        type=DatabaseType.MONGODB,
        host="localhost",
        port=27017,
        database="opensable",
        collection="documents"
    )
    
    print(f"MongoDB config: {mongo_config.type.value}")
    print(f"Database: {mongo_config.database}")
    print(f"Collection: {mongo_config.collection}")
    
    # Example 5: Redis operations (simulated)
    print("\n5. Redis Operations")
    print("-" * 40)
    
    redis_config = DatabaseConfig(
        type=DatabaseType.REDIS,
        host="localhost",
        port=6379,
        database="0"
    )
    
    print(f"Redis config: {redis_config.type.value}")
    print(f"Host: {redis_config.host}:{redis_config.port}")
    
    # Example 6: Query builder
    print("\n6. Query Builder")
    print("-" * 40)
    
    sqlite_manager = DatabaseManager(sqlite_config)
    await sqlite_manager.connect()
    
    # Build SELECT query
    query = sqlite_manager.query_builder()\
        .select("name", "email", "age")\
        .from_table("users")\
        .where("age > ?", 25)\
        .order_by("age", "DESC")\
        .limit(10)\
        .build()
    
    print(f"Built query: {query}")
    
    result = await sqlite_manager.query(
        "SELECT name, email, age FROM users WHERE age > ? ORDER BY age DESC LIMIT ?",
        params=(25, 10)
    )
    print(f"Results: {len(result.rows)} rows")
    for row in result.rows:
        print(f"  - {row}")
    
    await sqlite_manager.disconnect()
    
    # Example 7: Batch operations
    print("\n7. Batch Insert")
    print("-" * 40)
    
    sqlite_manager = DatabaseManager(sqlite_config)
    await sqlite_manager.connect()
    
    users_to_insert = [
        ("Eve", "eve@example.com", 27),
        ("Frank", "frank@example.com", 35),
        ("Grace", "grace@example.com", 29),
    ]
    
    for user in users_to_insert:
        await sqlite_manager.execute(
            "INSERT OR IGNORE INTO users (name, email, age) VALUES (?, ?, ?)",
            params=user
        )
    
    print(f"Batch inserted {len(users_to_insert)} users")
    
    result = await sqlite_manager.query("SELECT COUNT(*) as count FROM users")
    print(f"Total users in database: {result.rows[0]['count']}")
    
    await sqlite_manager.disconnect()
    
    print("\n" + "=" * 60)
    print("✅ Database examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
