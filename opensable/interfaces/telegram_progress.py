"""
Progress Bar Utilities for Telegram
Real-time progress indicators via message editing
"""
import asyncio
import logging
from typing import Optional, Callable, Any
from aiogram.types import Message
from datetime import datetime

logger = logging.getLogger(__name__)


class ProgressBar:
    """
    Real-time progress bar for long operations in Telegram.
    
    Features:
    - Visual progress bar (⬛⬜)
    - Percentage display
    - Time elapsed/remaining estimates
    - Auto-updates at configurable intervals
    - Supports async operations
    
    Usage:
        async with ProgressBar(message, "Downloading...") as progress:
            for i in range(100):
                await do_work(i)
                await progress.update(i, 100)
    """
    
    def __init__(
        self,
        message: Message,
        title: str = "Processing",
        bar_length: int = 10,
        update_interval: float = 0.5
    ):
        """
        Initialize progress bar.
        
        Args:
            message: Telegram message to edit
            title: Progress bar title
            bar_length: Length of progress bar in blocks
            update_interval: Minimum time between updates (seconds)
        """
        self.message = message
        self.title = title
        self.bar_length = bar_length
        self.update_interval = update_interval
        
        self.progress_message: Optional[Message] = None
        self.start_time = None
        self.last_update = 0
        self.current = 0
        self.total = 100
    
    async def __aenter__(self):
        """Start progress tracking"""
        self.start_time = datetime.now()
        self.progress_message = await self.message.answer(
            self._format_progress(0, 100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Complete progress tracking"""
        if self.progress_message:
            # Show 100% complete
            await self.update(self.total, self.total, force=True)
            await asyncio.sleep(1)  # Show completion briefly
    
    async def update(
        self,
        current: int,
        total: int = None,
        status: str = None,
        force: bool = False
    ):
        """
        Update progress.
        
        Args:
            current: Current progress value
            total: Total value (if changed)
            status: Optional status message
            force: Force update even if interval not reached
        """
        if total is not None:
            self.total = total
        
        self.current = current
        
        # Rate limiting
        now = asyncio.get_event_loop().time()
        if not force and (now - self.last_update) < self.update_interval:
            return
        
        self.last_update = now
        
        # Update message
        if self.progress_message:
            try:
                text = self._format_progress(current, self.total, status)
                await self.progress_message.edit_text(text)
            except Exception as e:
                # Ignore telegram rate limits
                if "message is not modified" not in str(e).lower():
                    logger.debug(f"Progress update failed: {e}")
    
    def _format_progress(
        self,
        current: int,
        total: int,
        status: str = None
    ) -> str:
        """Format progress bar text"""
        # Calculate percentage
        if total > 0:
            percent = int((current / total) * 100)
        else:
            percent = 0
        
        # Build progress bar
        filled = int((current / total) * self.bar_length) if total > 0 else 0
        bar = "⬛" * filled + "⬜" * (self.bar_length - filled)
        
        # Time estimates
        time_info = ""
        if self.start_time and percent > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if percent < 100:
                estimated_total = (elapsed / percent) * 100
                remaining = estimated_total - elapsed
                time_info = f"\n⏱️ {self._format_time(remaining)} remaining"
            else:
                time_info = f"\n✅ Completed in {self._format_time(elapsed)}"
        
        # Build message
        text = f"**{self.title}**\n\n"
        text += f"{bar} {percent}%"
        if status:
            text += f"\n📝 {status}"
        text += time_info
        
        return text
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"


async def with_progress(
    message: Message,
    async_func: Callable,
    title: str = "Processing",
    *args,
    **kwargs
) -> Any:
    """
    Execute async function with progress bar.
    
    The async function should accept a 'progress' callback parameter:
        async def my_task(data, progress=None):
            for i, item in enumerate(data):
                await process(item)
                if progress:
                    await progress(i, len(data))
    
    Usage:
        result = await with_progress(
            message,
            download_file,
            title="Downloading PDF",
            url="https://..."
        )
    """
    progress_bar = ProgressBar(message, title)
    
    async def progress_callback(current, total, status=None):
        await progress_bar.update(current, total, status)
    
    async with progress_bar:
        # Inject progress callback
        kwargs['progress'] = progress_callback
        result = await async_func(*args, **kwargs)
    
    return result


class Spinner:
    """
    Simple spinner for indeterminate progress.
    
    Shows rotating animation: ⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏
    """
    
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(
        self,
        message: Message,
        title: str = "Processing",
        update_interval: float = 0.1
    ):
        self.message = message
        self.title = title
        self.update_interval = update_interval
        self.spinner_message: Optional[Message] = None
        self.frame_index = 0
        self.running = False
        self._task = None
    
    async def __aenter__(self):
        """Start spinner"""
        self.running = True
        self.spinner_message = await self.message.answer(
            f"{self.FRAMES[0]} {self.title}..."
        )
        self._task = asyncio.create_task(self._animate())
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop spinner"""
        self.running = False
        if self._task:
            await self._task
        
        if self.spinner_message:
            try:
                await self.spinner_message.edit_text(f"✅ {self.title} complete")
            except:
                pass
    
    async def _animate(self):
        """Animate spinner frames"""
        while self.running:
            try:
                self.frame_index = (self.frame_index + 1) % len(self.FRAMES)
                frame = self.FRAMES[self.frame_index]
                
                if self.spinner_message:
                    await self.spinner_message.edit_text(
                        f"{frame} {self.title}..."
                    )
                
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                if "message is not modified" not in str(e).lower():
                    logger.debug(f"Spinner animation failed: {e}")
                await asyncio.sleep(self.update_interval)


# Example usage
async def example_long_task(data, progress=None):
    """Example task with progress tracking"""
    total = len(data)
    
    for i, item in enumerate(data):
        # Simulate work
        await asyncio.sleep(0.1)
        
        # Update progress
        if progress:
            await progress(i + 1, total, status=f"Processing {item}")
    
    return "Done!"


async def example_usage(message: Message):
    """Example usage of progress utilities"""
    
    # Progress bar example
    async with ProgressBar(message, "Downloading file") as progress:
        for i in range(100):
            await asyncio.sleep(0.05)
            await progress.update(i + 1, 100)
    
    # Spinner example
    async with Spinner(message, "Generating response"):
        await asyncio.sleep(3)  # Simulate work
    
    # with_progress helper
    data = [f"item_{i}" for i in range(50)]
    result = await with_progress(
        message,
        example_long_task,
        title="Processing items",
        data=data
    )
