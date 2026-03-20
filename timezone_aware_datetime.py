import datetime
import time
from typing import Optional, Union, Callable
from zoneinfo import ZoneInfo

class TimezoneAwareError(Exception):
    """Custom exception for timezone-related errors."""
    pass

def get_local_timezone() -> ZoneInfo:
    """Detect the system's local timezone automatically."""
    try:
        # Use the system's TZ name from time module
        tz_name = time.tzname[0]
        # Try /etc/timezone first (Linux)
        try:
            with open("/etc/timezone", "r") as f:
                tz_name = f.read().strip()
                return ZoneInfo(tz_name)
        except (FileNotFoundError, KeyError):
            pass
        # Try /etc/localtime symlink
        import os
        try:
            link = os.readlink("/etc/localtime")
            # e.g. /usr/share/zoneinfo/America/New_York
            tz_name = "/".join(link.split("/zoneinfo/")[1:])
            if tz_name:
                return ZoneInfo(tz_name)
        except (OSError, IndexError, KeyError):
            pass
        # Fallback to UTC
        return ZoneInfo("UTC")
    except Exception as e:
        raise TimezoneAwareError(f"Could not detect local timezone: {e}") from e


from dataclasses import dataclass
import logging

def logger():
    return logging.getLogger(__name__)

@dataclass
class ConversionContext:
    """
    Contextual information for timezone conversion.
    
    Attributes:
        original_dt: The input datetime before conversion
        local_tz: Detected local timezone
        target_zi: Target ZoneInfo object
        operation_id: Unique identifier for tracking conversions
    """
    original_dt: Optional[datetime.datetime] = None
    local_tz: Optional[str] = None
    target_zi: Optional[ZoneInfo] = None


def convert_timezone(dt: datetime.datetime, target_tz: str) -> datetime.datetime:
    """
    Convert a datetime to a target timezone, ensuring timezone awareness.
    
    Args:
        dt: Input datetime (can be naive or aware)
        target_tz: Target timezone string (e.g., 'America/New_York')
    
    Returns:
        datetime.datetime: Datetime in target timezone
    
    Raises:
        TimezoneAwareError: If conversion fails
    """
    from zoneinfo import ZoneInfo

    # Determine if dt is naive (no timezone info)
    is_naive = dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None

    if is_naive:
        # If no timezone specified, use local timezone
        if target_tz is None:
            target_tz = get_local_timezone().key

        # Parse target timezone
        try:
            target_zi = ZoneInfo(target_tz)
        except Exception as e:
            raise TimezoneAwareError(f"Invalid timezone: {target_tz}") from e

        # If dt is naive but we have a target timezone, localize
        try:
            # If dt already has timezone (could be fold situation)
            if dt.tzinfo:
                dt = dt.replace(tzinfo=target_zi)
            else:
                dt = dt.replace(tzinfo=target_zi)
        except Exception as e:
            raise TimezoneAwareError(f"Failed to localize datetime: {dt}") from e

    # Convert to target timezone
    try:
        dt = dt.astimezone(ZoneInfo(target_tz))
    except Exception as e:
        raise TimezoneAwareError(f"Failed to convert timezone: {dt}") from e

    return dt
    """
    Convert a datetime to a target timezone.
    
    Args:
        dt: Input datetime (can be naive or aware)
        target_tz: Target timezone name (e.g., 'America/New_York')
    
    Returns:
        datetime.datetime: Datetime in the target timezone
    
    Raises:
        TimezoneAwareError: If conversion fails
    """
    if not isinstance(dt, datetime.datetime):
        raise TimezoneAwareError(f"Expected datetime, got {type(dt).__name__}")
    
    try:
        original_copy = dt.replace(tzinfo=None)  # Create copy without timezone info
        
        if dt.tzinfo is None:
            # Naive datetime - assume it's local time
            local_tz = get_local_timezone()
            dt = dt.replace(tzinfo=local_tz)
        
        target_zi = ZoneInfo(target_tz)
        return dt.astimezone(target_zi)
    except Exception as e:
        raise TimezoneAwareError(f"Timezone conversion failed: {e}") from e

def now(tz: Optional[str] = None) -> datetime.datetime:
    """
    Get the current time in a specified timezone, or local time if none is provided.
    
    Args:
        tz: Timezone name (e.g., 'UTC', 'Europe/London')
    
    Returns:
        datetime.datetime: Current time
    """
    try:
        if tz:
            return datetime.datetime.now(ZoneInfo(tz))
        return datetime.datetime.now().replace(tzinfo=get_local_timezone())
    except Exception as e:
        raise TimezoneAwareError(f"Failed to get current time: {e}") from e

def parse_datetime(s: str) -> datetime.datetime:
    """
    Parse a datetime string, automatically detecting the format.
    Supports common formats including timezone-aware parsing.
    
    Args:
        s: Datetime string to parse
    
    Returns:
        datetime.datetime: Parsed datetime object
    
    Raises:
        TimezoneAwareError: If parsing fails
    """
    try:
        # Try ISO 8601 format first (most reliable)
        return datetime.datetime.fromisoformat(s)
    except ValueError:
        pass
    
    # Try common date string formats
    formats = [
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    
    raise TimezoneAwareError(f"Could not parse datetime string: {s}")

def ensure_timezone(dt: datetime.datetime) -> datetime.datetime:
    """
    Ensure a datetime has a timezone. If it's naive, use local timezone.
    
    Args:
        dt: Datetime to check
    
    Returns:
        datetime.datetime: Timezone-aware datetime
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=get_local_timezone())
    return dt

def compare_datetimes(dt1: datetime.datetime, dt2: datetime.datetime) -> int:
    """
    Compare two datetimes safely, handling timezone differences by converting
    both to UTC for comparison.
    
    Args:
        dt1: First datetime
        dt2: Second datetime
    
    Returns:
        int: -1 if dt1 < dt2, 0 if equal, 1 if dt1 > dt2
    """
    try:
        # Convert both to UTC for fair comparison
        utc_dt1 = convert_to_timezone(dt1, 'UTC')
        utc_dt2 = convert_to_timezone(dt2, 'UTC')
        
        if utc_dt1 < utc_dt2:
            return -1
        elif utc_dt1 == utc_dt2:
            return 0
        else:
            return 1
    except Exception as e:
        raise TimezoneAwareError(f"Datetime comparison failed: {e}") from e

def tz_aware_decorator(func: Callable) -> Callable:
    """
    Decorator to automatically make function's return datetime values timezone-aware.
    
    Args:
        func: Function to decorate
    
    Returns:
        Callable: Decorated function
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, datetime.datetime):
            return ensure_timezone(result)
        elif isinstance(result, list):
            return [ensure_timezone(d) if isinstance(d, datetime.datetime) else d for d in result]
        return result
    
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    return wrapper

# Example usage
if __name__ == "__main__":
    # Create some sample datetimes
    now_utc = now('UTC')
    now_local = now()
    
    print(f"Current UTC: {now_utc}")
    print(f"Current Local: {now_local}")
    
    # Demonstrate conversion
    sample_dt = datetime.datetime(2023, 10, 15, 12, 30, 0)
    print(f"Original (naive): {sample_dt}")
    print(f"Converted to Europe/London: {convert_to_timezone(sample_dt, 'Europe/London')}")
    
    # Demonstrate comparison
    dt1 = convert_to_timezone(datetime.datetime(2023, 10, 15, 12, 0), 'America/New_York')
    dt2 = convert_to_timezone(datetime.datetime(2023, 10, 15, 17, 0), 'Europe/London')
    
    print(f"Comparison of {dt1} and {dt2}: {compare_datetimes(dt1, dt2)}")
