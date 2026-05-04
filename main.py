"""
main.py,  Sable entry point wrapper.
The real implementation lives in opensable/__main__.py
Run with: python main.py  OR  python -m opensable
"""
from opensable.__main__ import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
